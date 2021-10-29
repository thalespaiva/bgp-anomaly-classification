#!/usr/bin/env python3

import argparse
import datetime as dt
import pandas as pd
import geopy.distance
import numpy as np
import os
import sys
import pytz

from csv import reader
from dateutil.relativedelta import relativedelta
from functools import wraps
from pandas.core.indexing import check_bool_indexer
from scipy import stats
from io import StringIO
from pprint import pprint

from relationships_inference import ASGraph


# NOTICE: Differently from bgpreader, pybgpstream does not consider the origin_AS field in its
# dump. Therefore, if you used bgpreader for the dump, you may need to add the origin_AS field
# in the appropriage place in the global list below.
COLUMNS = [
    'rec_type',
    'elem_type',
    'timestamp',
    'project',
    'collector',
    'router',
    'router_ip',
    'peer_ASN',
    'peer_IP',
    'prefix',
    'next_hop_IP',
    'AS_path',
    # Be careful if your dump (e.g. from bgpreader) considers the origin_AS field.
    'communities',
    'old_state',
    'new_state',
    'label',
    # COMPUTED FIELD: 'collapsed_as_path' - KEEP THIS LINE COMMENTED
    # COMPUTED FIELD: 'origin_AS' - KEEP THIS LINE COMMENTED
]


TIMESTAMP_COLUMN = COLUMNS.index('timestamp')


def parse_unix_timestamp(str_timestamp):
    try:
        sec_str, usec_str = str_timestamp.split('.')
        return (dt.datetime.fromtimestamp(int(sec_str), tz=dt.timezone.utc) +
                dt.timedelta(microseconds=int(usec_str)))
    except ValueError:
        return dt.datetime.fromtimestamp(int(str_timestamp), tz=dt.timezone.utc)


def get_first_day_of_previous_month(date: dt.datetime):
    return (date - relativedelta(months=1)).replace(day=1).replace(hour=0, minute=0, second=0)


def read_csv(filepath_or_strio, **pandas_opts):
    df = pd.read_csv(
        filepath_or_strio,
        delimiter='|',
        header=None,
        names=COLUMNS,
        converters={
            'timestamp': parse_unix_timestamp,
            'AS_path': lambda x: parse_path(x, not_collapse_prepending_asns=True)
        },
        **pandas_opts)

    df['collapsed_as_path'] = df.apply(lambda row: collapse_path(row.AS_path), axis=1)
    df['origin_AS'] = df.apply(lambda row: row.collapsed_as_path[-1], axis=1)

    return df

def parse_path_base(str_path):
    try:
        clean = str_path.replace('{', ' ') \
                        .replace('}', ' ') \
                        .replace(',', ' ')

        return list(clean.split())
    except ValueError:
        return []


def collapse_path(path):
    path_clean = [path[0]]
    for u in path[1:]:
        if u != path_clean[-1]:
            path_clean.append(u)
    return path_clean


def parse_path(path_str, not_collapse_prepending_asns=False):
    path = parse_path_base(path_str)

    if not_collapse_prepending_asns:
        return path

    return collapse_path(path)


# Even though this class should be used as a decorator, I the class naming convention for
# its names. Apparently there are no explicit naming conventions for this case but I may be wrong.
#
# This class is based on an answer in the following stackoverflow post:
# https://stackoverflow.com/questions/5910703/how-to-get-all-methods-of-a-python-class-with-given-decorator
#
class ComputedFeature:
    """
    This class is to be used as a decorator for feature that should be computed by a Feature
    Extractor. The Feature Extractor will then have access to the decorated methods and
    compute only those registered features.
    """

    def __init__(self, method):
        self._method = method

    def __call__(self, obj, *args, **kwargs):
        return {self._method.__name__: self._method(obj, *args, **kwargs)}

    @classmethod
    def methods(cls, subject):
        def get_annotated_methods():
            for name in dir(subject):
                method = getattr(subject, name)
                if isinstance(method, ComputedFeature):
                    yield name, method

        return {name: method for name, method in get_annotated_methods()}


class ComputedMultiFeature(ComputedFeature):

    def __call__(self, obj, *args, **kwargs):
        return self._method(obj, *args, **kwargs)


class PrecomputedModelCollection:
    """
    This class provides an interface for the FeatureExtractor to access
    the precomputed results, which should be stored in a directory.
    By precomputed results, we currently mean:
        * AS graph (with relationships)
        * Trained BGP2Vec models

    TODO: Use BGP2Vec models.
    TODO: Write explicit requirements on the filenames.
    """
    def __init__(self, dirpath):
        self.dirpath = dirpath

        # self.bgp2vecs = self._init_bgp2vec_models()
        # self.as_graphs = self._init_as_graphs_models()
        # self.as_relationships = self._init_as_relationships_models()

        self.as_graph_cache = {}

    def get_index_date_for_chunk(self, chunk_array):

        # If chunk_array is empty, an IndexError will be raised.
        first_timestamp = chunk_array.timestamp[0]
        return get_first_day_of_previous_month(first_timestamp)

    def compute_as_graph_for_chunk(self, chunk_array):
        index_date = self.get_index_date_for_chunk(chunk_array)

        fname = index_date.strftime('%Y-%m-%d_%H:%M:%S') + '.paths'
        fpath = os.path.join(self.dirpath, 'paths', fname)

        if not os.path.isfile(fpath):
            raise FileNotFoundError(f'Could not find file {fpath} containing paths. Please \
                                      be sure that you are using the correct path and have already \
                                      downloaded the paths from RIBs.',)

        return ASGraph(fpath)

    def cache_as_graph(self, as_graph, index_date):
        self.as_graph_cache = {
            index_date: as_graph
        }

    def _get_as_graph_for_date_in_cache(self, index_date):
        return self.as_graph_cache[index_date]

    def get_as_graph_for_chunk(self, chunk_array):

        index_date = self.get_index_date_for_chunk(chunk_array)

        try:
            return self._get_as_graph_for_date_in_cache(index_date)
        except KeyError:
            as_graph = self.compute_as_graph_for_chunk(chunk_array)
            self.cache_as_graph(as_graph, index_date)

            return as_graph


def apply_for_announcements_only(f):
    @wraps(f)
    def dec_f(self, array):
        return f(self, array[array.elem_type == 'A'])
    return dec_f


class FeatureExtractor:

    def __init__(self,
                 precomputed_model_collection_dir: str,
                 zero_day_file: str,
                 verbose=False):

        self.precomputed = PrecomputedModelCollection(precomputed_model_collection_dir)

        self.verbose = verbose
        self.asn_geo_state = {}
        self.asn_country_map = self.asn_countries_dataset_to_map()
        self.countries_dists = {}
        self.zero_day = self.init_asn_geo_state(zero_day_file)

    def get_names_of_features_to_be_computed(self):
        null_df = pd.DataFrame(columns=COLUMNS)
        features = self.compute_features(null_df)
        return list(features.keys())

    def get_features_header(self, use_filename_field=False):
        names_of_features = self.get_names_of_features_to_be_computed()
        return ','.join(names_of_features)

    def extract_features_from_file(self, filepath=None, timedelta=dt.timedelta(minutes=1)):
        def get_time_from_line(l):
            return parse_unix_timestamp(l.split('|')[TIMESTAMP_COLUMN])

        if filepath is None:
            file = sys.stdin
        else:
            file = open(filepath, 'r')

        chunk_first_line = file.readline()
        start_time = get_time_from_line(chunk_first_line).replace(second=0, microsecond=0)

        names_of_features = self.get_names_of_features_to_be_computed()
        print('time,' + ','.join(names_of_features))

        i = 0
        while chunk_first_line:
            i += 1

            chunk = [chunk_first_line]
            chunk_start_time = start_time + (i - 1)*timedelta

            next_chunk_first_line = None
            for line in file:
                sample_time = get_time_from_line(line)
                if (sample_time - start_time) < i*timedelta:
                    chunk.append(line)
                    continue
                else:
                    # This prepares the initialization of the next chunk
                    next_chunk_first_line = line
                    break

            # Build chunk complete
            try:
                features = self.extract_features_from_chunk(chunk)

                str_chunk_time = chunk_start_time.strftime('%Y-%m-%d %H:%M:%S')
                print(f'{str_chunk_time},' + ','.join(map(str, features.values())))
            except Exception as e:
                print('FAILED TO READ CHUNK: =========', file=sys.stderr)
                print(e, file=sys.stderr)
                pprint(chunk, sys.stderr)
                print('END OF PROBLEMATIC CHUNK =========')
                raise e

            chunk_first_line = next_chunk_first_line

    def extract_features_from_chunk(self, chunk):
        strio_chunk = StringIO(''.join(chunk))
        df = read_csv(strio_chunk)

        if self.verbose:
            print('============ CHUNK DATAFRAME ===============', file=sys.stderr)
            print(df, file=sys.stderr)
            print('============================================', file=sys.stderr)
        return self.compute_features(df)

    def compute_features(self, array):
        features_to_compute = ComputedFeature.methods(FeatureExtractor)
        features = {}

        for fname, f in features_to_compute.items():
            features.update(f(self, array))

        return features

    @ComputedFeature
    def label(self, array):
        if array.empty:
            return pd.NA

        return stats.mode(array.label)[0][0]

    # BEGIN VOLUME-BASED FEATURES

    @ComputedMultiFeature
    def n_message_types(self, array):
        message_type_counts = {
            'A': 0,
            'W': 0,
            'S': 0,
        }

        for i, row in array.iterrows():
            message_type_counts[row.elem_type] += 1

        return {
            'n_announcements': message_type_counts['A'],
            'n_withdrawals': message_type_counts['W'],
            'n_states': message_type_counts['S'],
        }

    # END VOLUME-BASED FEATURES

    # BEGIN GRAPH-BASED FEATURES

    @ComputedMultiFeature
    @apply_for_announcements_only
    def av_number_of_edges_of_each_type(self, array):

        output = {
            etype: 0 for etype in ['P2P', 'C2P', 'P2C', 'S2S']
        }
        n_edges = 0

        try:
            as_graph = self.precomputed.get_as_graph_for_chunk(array)
            for path in array.collapsed_as_path:
                for i, u in enumerate(path[:-1]):
                    u_next = path[i + 1]

                    etype = as_graph.get_relationship(u, u_next)
                    if not etype:
                        continue
                    else:
                        output[etype] += 1

                    n_edges += 1

        except (IndexError, KeyError):
            return {
                f'av_number_of_{etype}_edges': 0 for etype in output
            }

        return {
            f'av_number_of_{etype}_edges': (n/n_edges if n_edges > 0 else 0)
                for etype, n in output.items()
        }

    @ComputedMultiFeature
    @apply_for_announcements_only
    def av_and_var_of_as_degree_in_paths(self, array):

        try:
            as_graph = self.precomputed.get_as_graph_for_chunk(array)
            degrees = []
            for path in array.collapsed_as_path:
                for u in path:
                    degrees.append(len(as_graph.neighbors[u]))

            var_as_degree_in_paths = np.var(degrees)
            av_as_degree_in_paths = np.mean(degrees)


        except (KeyError, IndexError):
            var_as_degree_in_paths = pd.NA
            av_as_degree_in_paths = pd.NA

        return {
            'var_as_degree_in_paths': var_as_degree_in_paths,
            'av_as_degree_in_paths': av_as_degree_in_paths,
            # TODO: Consider degree-based valleys
        }

    @ComputedFeature
    @apply_for_announcements_only
    def av_number_of_non_vf_paths(self, array):
        count = 0
        try:
            as_graph = self.precomputed.get_as_graph_for_chunk(array)
            for path in array.collapsed_as_path:
                try:
                    if not as_graph.is_valley_free(path):
                        count += 1
                except KeyError:
                    pass

        except (IndexError, KeyError):
            return pd.NA

        return count/len(array)

    @ComputedFeature
    @apply_for_announcements_only
    def av_number_of_edges_not_in_as_graph(self, array):

        number_of_edges_not_in_as_graph = 0

        try:
            as_graph = self.precomputed.get_as_graph_for_chunk(array)


            for path in array.AS_path:
                for i, u in enumerate(path[:-1]):
                    u_next = path[i + 1]

                    if u not in as_graph.neighbors or u_next not in as_graph.neighbors[u]:
                        number_of_edges_not_in_as_graph += 1

        except (IndexError, KeyError):
            return pd.NA

        return number_of_edges_not_in_as_graph/len(array)

    # END GRAPH-BASED FEATURES

    # BEGIN PATH-BASED FEATURES

    @ComputedMultiFeature
    @apply_for_announcements_only
    def av_and_max_unique_as_path_length(self, array):

        try:
            lens = [len(path) for path in array.collapsed_as_path]

            max_as_path_length = np.max(lens)
            av_as_path_length = np.mean(lens)

        except (ValueError, AttributeError):
            max_as_path_length = pd.NA
            av_as_path_length = pd.NA

        return {
            'max_unique_as_path_length': max_as_path_length,
            'av_unique_as_path_length': av_as_path_length,
        }

    @ComputedMultiFeature
    @apply_for_announcements_only
    def av_and_max_as_path_length(self, array):
        lens = [len(path) for path in array.AS_path]

        try:
            max_as_path_length = np.max(lens)
            av_as_path_length = np.mean(lens)

        except ValueError:
            max_as_path_length = pd.NA
            av_as_path_length = pd.NA

        return {
            'max_as_path_length': max_as_path_length,
            'av_as_path_length': av_as_path_length,
        }

    # END PATH-BASED FEATURES

    # BEGIN PREFIX-BASED FEATURES

    @ComputedMultiFeature
    @apply_for_announcements_only
    def av_and_max_number_of_bits_in_prefix(self, array):

        stats = {
            ip_version: {
                'sum' : 0,
                'total': 0,
                'max': 0,
            }
            for ip_version in ['ipv4', 'ipv6']
        }

        def get_ip_version(base_ip):
            if base_ip.count('.') == 3:
                return 'ipv4'
            elif ':' in base_ip:
                return 'ipv6'
            return None

        for pref in array.prefix:
            try:
                base_ip, nbits = pref.split('/')

                ip_version = get_ip_version(base_ip)

                stats[ip_version]['sum'] += int(nbits)
                stats[ip_version]['max'] = max(stats[ip_version]['max'], int(nbits))
                stats[ip_version]['total'] += 1

            except ValueError:
                continue

        if stats['ipv4']['total'] > 0:
            av_number_of_bits_in_prefix_ipv4 = stats['ipv4']['sum']/stats['ipv4']['total']
            max_number_of_bits_in_prefix_ipv4 = stats['ipv4']['max']
        else:
            av_number_of_bits_in_prefix_ipv4 = pd.NA
            max_number_of_bits_in_prefix_ipv4 = pd.NA


        if stats['ipv6']['total'] > 0:
            av_number_of_bits_in_prefix_ipv6 = stats['ipv6']['sum']/stats['ipv6']['total']
            max_number_of_bits_in_prefix_ipv6 = stats['ipv6']['max']
        else:
            av_number_of_bits_in_prefix_ipv6 = pd.NA
            max_number_of_bits_in_prefix_ipv6 = pd.NA

        return {
            'av_number_of_bits_in_prefix_ipv4': av_number_of_bits_in_prefix_ipv4,
            'max_number_of_bits_in_prefix_ipv4': max_number_of_bits_in_prefix_ipv4,
            'av_number_of_bits_in_prefix_ipv6': av_number_of_bits_in_prefix_ipv6,
            'max_number_of_bits_in_prefix_ipv6': max_number_of_bits_in_prefix_ipv6,
        }

    # END PREFIX-BASED FEATURES

    # BEGIN LOCATION-BASED FEATURES

    @ComputedMultiFeature
    @apply_for_announcements_only
    def avg_asn_geo_dists(self, array):
        # TODO: We need to make sure that this feature does not leak information
        # on WHEN data was collected with respect to the beginning of the event.
        # For example: since we start collecting two days before the event, and
        # no prefix was seen, this makes the values of the features close to 0,
        # simply because they were the first rows.
        
        i = 0

        total_same_bitlen = 0
        total_items_same_bitlen = 0
        total_diff_bitlen = 0
        total_items_diff_bitlen = 0
        
        previous_date = dt.datetime.utcnow().replace(tzinfo=pytz.utc)
        if len(array) > 0:
            previous_date = array.timestamp.iloc[0]
        self.asn_geo_state[previous_date.day] = {}

        the_day_before = self.zero_day
        for item in array.prefix:
            current_date = array.timestamp.iloc[i]
            if current_date.day != previous_date.day:
                #changed day
                #delete the day before previous from dict
                del self.asn_geo_state[the_day_before]
                the_day_before = previous_date.day
                if the_day_before not in self.asn_geo_state:
                    self.asn_geo_state[the_day_before] = {}
                self.asn_geo_state[current_date.day] = {}

            prefix_split = item.split('/')
            prefix_address = prefix_split[0]
            prefix_bitlen = prefix_split[1] if len(prefix_split) > 1 else '_' 

            if prefix_address in self.asn_geo_state[current_date.day] or \
                prefix_address in self.asn_geo_state[the_day_before]:

                key = the_day_before
                if prefix_address in self.asn_geo_state[current_date.day]:
                    key = current_date.day
                else:
                    self.asn_geo_state[current_date.day][prefix_address] = {}
                if prefix_bitlen in self.asn_geo_state[key][prefix_address] \
                    and self.asn_geo_state[key][prefix_address][prefix_bitlen] != array.origin_AS.iloc[i]:

                    dist = self.compute_geo_dist_asn(
                            self.asn_geo_state[key][prefix_address][prefix_bitlen], 
                            array.origin_AS.iloc[i])

                    total_same_bitlen += dist
                    #AVG is being calculated counting 0 values
                    #This can be changed by uncommenting the line below
                    #if dist > 0:
                    total_items_same_bitlen += 1
                if self.asn_geo_state[key][prefix_address]['last_bitlen'] != prefix_bitlen:
                    if array.origin_AS.iloc[i] != self.asn_geo_state[key][prefix_address][
                        self.asn_geo_state[key][prefix_address]['last_bitlen']]:
                        dist = self.compute_geo_dist_asn(
                            self.asn_geo_state[key][prefix_address]
                                [self.asn_geo_state[key][prefix_address]['last_bitlen']], 
                            array.origin_AS.iloc[i]
                            )
                        total_diff_bitlen += dist
                        #AVG is being calculated counting 0 values
                        #This can be changed by uncommenting the line below
                        #if dist > 0:
                        total_items_diff_bitlen += 1
            else:
                self.asn_geo_state[current_date.day][prefix_address] = {}
            self.asn_geo_state[current_date.day][prefix_address][prefix_bitlen] = array.origin_AS.iloc[i]
            self.asn_geo_state[current_date.day][prefix_address]['last_bitlen'] = prefix_bitlen
            i += 1
            previous_date = current_date

        return {
            'avg_geo_dist_same_bitlen' : total_same_bitlen/total_items_same_bitlen if total_items_same_bitlen > 0 else 0,
            'avg_geo_dist_diff_bitlen' : total_diff_bitlen/total_items_diff_bitlen if total_items_diff_bitlen > 0 else 0
        }
        
    # BEGIN HELPER FUNCTIONS
    def compute_geo_dist_asn(self, asn1, asn2):
        asn1 = str(asn1)
        asn2 = str(asn2)
        row1 = self.asn_country_map[asn1] if asn1 in self.asn_country_map else None
        row2 = self.asn_country_map[asn2] if asn2 in self.asn_country_map else None

        if row1 and row2:
            if row1['country'] != row2['country']:
                sorted_countries = sorted([row1['country'], row2['country']])
                asn_key = sorted_countries[0] + '_' + sorted_countries[1]
                
                if asn_key in self.countries_dists:
                    return self.countries_dists[asn_key]

                coords_1 = (row1['latitude'], row1['longitude'])
                coords_2 = (row2['latitude'], row2['longitude'])

                dist = geopy.distance.geodesic(coords_1, coords_2).km
                self.countries_dists[asn_key] = dist
                return dist
        return 0

    def asn_countries_dataset_to_map(self):
        ans = {}
        with open('asn_countries.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            for row in csv_reader:
                ans[row[0].strip()] = {
                    'country': row[1].strip(),
                    'latitude': float(row[2].strip()),
                    'longitude': float(row[3].strip())
                }
        return ans
    
    def init_asn_geo_state(self, zero_day_file):
        def get_time_from_line(l):
            return parse_unix_timestamp(l.split('|')[TIMESTAMP_COLUMN])
        
        if zero_day_file is None:
            file = sys.stdin
        else:
            file = open(zero_day_file, 'r')

        chunk_first_line = file.readline()
        start_time = get_time_from_line(chunk_first_line).replace(second=0, microsecond=0)
        zero_day = start_time.day

        self.asn_geo_state = {}
        self.asn_geo_state[zero_day] = {}

        i = 0
        while chunk_first_line:
            item = chunk_first_line.split('|')
            #Columns order:
            #'rec_type','elem_type','timestamp','project','collector',
            #'router','router_ip','peer_ASN','peer_IP','prefix','next_hop_IP',
            #'AS_path', # Be careful if your dump (e.g. from bgpreader) considers the origin_AS field.
            #'communities','old_state','new_state','label'

            item[11] = parse_path(item[11], not_collapse_prepending_asns=True)
            collapsed_as_path = collapse_path(item[11])
            origin_as = collapsed_as_path[-1]
            
            prefix_split = item[9].split('/')
            prefix_address = prefix_split[0]
            prefix_bitlen = prefix_split[1] if len(prefix_split) > 1 else '_' 

            if origin_as is not None and origin_as != 'None':
                if prefix_address not in self.asn_geo_state[zero_day]:
                    self.asn_geo_state[zero_day][prefix_address] = {}

                self.asn_geo_state[zero_day][prefix_address][prefix_bitlen] = origin_as
                self.asn_geo_state[zero_day][prefix_address]['last_bitlen'] = prefix_bitlen
            
            i += 1
            chunk_first_line = file.readline()

        return zero_day
        
    # END HELPER FUNCTIONS
    # END LOCATION-BASED FEATURES


def main(args):
    fx = FeatureExtractor(args.precomputed_dir, args.zero_day_file, verbose=args.verbose)
    fx.extract_features_from_file(filepath=args.input, timedelta=args.interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to file storing BGP update messages.')
    parser.add_argument('interval',
                        type=lambda x: dt.timedelta(minutes=int(x)),
                        help='Time interval for each chunk in minutes.')
    parser.add_argument('precomputed_dir', help='Directory of the precomputed models.')
    parser.add_argument('zero_day_file', help='Path to file storing BGP update messages of the day before the event start date.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Set verbose output.')
    args = parser.parse_args()

    main(args)

