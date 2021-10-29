#!/usr/bin/env python3

import argparse
import datetime as dt
import itertools as it
import os
import pandas as pd
import pybgpstream
import pytz
import sys

from dateutil.relativedelta import relativedelta
from tqdm import tqdm


NUMBERS_OF_DAYS_TO_COLLECT = 7

COLLECTORS = [
    'rrc04',
    'rrc05',
    'rrc06',
]

def parse_path(path_str, not_collapse_prepending_asns=False):
    if not_collapse_prepending_asns:
        return path_str

    path = path_str.split(' ')

    path_clean = [path[0]]
    for u in path[1:]:
        if u != path_clean[-1]:
            path_clean.append(u)

    path_clean_str = ' '.join(path_clean)

    return path_clean_str


def collect_paths(base_datetime, ndays, fname, args):

    days_collectors = it.product(range(ndays), COLLECTORS)
    tqdm_total = ndays*len(COLLECTORS)

    unique_paths = set()

    foutput = open(os.path.join(args.directory, fname), 'w')

    for nday in tqdm(range(ndays), desc='Days'):

        date_time = base_datetime + dt.timedelta(days=nday)
        date_time_str = date_time.strftime(r'%Y-%m-%d %H:%M:%S')

        until_time_str = (date_time + dt.timedelta(hours=2)).strftime(r'%Y-%m-%d %H:%M:%S')
        print(f'Collecting data from {COLLECTORS} at {date_time_str} to {until_time_str}', file=sys.stderr)

        stream = pybgpstream.BGPStream(
            from_time=date_time_str, until_time=until_time_str,
            collectors=COLLECTORS,
            record_type="ribs",
        )

        n_paths_for_pair = 0

        for elem in stream:
            path_str = elem.fields["as-path"]
            path_clean_str = parse_path(path_str, args.not_collapse_prepending_asns)

            if args.verbose and path_clean_str != path_str:
                print(f'Cleaned {path_str} to {path_clean_str}', file=sys.stderr)

            if not args.not_only_unique_paths:
                if path_clean_str in unique_paths:
                    if args.verbose:
                        print(f'Found repeated path {path_clean_str}', file=sys.stderr)

                    continue

                unique_paths.add(path_clean_str)

            elem_time = dt.datetime.fromtimestamp(elem.time)
            if args.not_path_only:
                print(f'{elem_time}|{elem.collector}|{path_clean_str}', file=foutput)
            else:
                print(f'{path_clean_str}', file=foutput)

            n_paths_for_pair += 1

        print(f'Added {n_paths_for_pair} paths', file=sys.stderr)

    foutput.close()

def read_anomalies_csv(events_file):
    def parse_date(x):
        return pytz.utc.localize(dt.datetime.fromisoformat(x))

    return pd.read_csv(
        events_file,
        converters={
            'start_date': parse_date,
            'finish_date': parse_date,
        })


def get_first_day_of_previous_month(date):
    return (date - relativedelta(months=1)).replace(day=1).replace(hour=0, minute=0, second=0)


def main(args):
    events_df = read_anomalies_csv(args.events_file)

    for i, event_data in events_df.iterrows():
        print(i, event_data)

        base_datetime = get_first_day_of_previous_month(event_data.start_date)

        fname = base_datetime.strftime('%Y-%m-%d_%H:%M:%S') + '.paths'
        print(f'File: {fname}')
        collect_paths(base_datetime, NUMBERS_OF_DAYS_TO_COLLECT, fname, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_file', help='CSV file with containing the information of the \
                         events to consider.')
    parser.add_argument('directory', help='Directory in which to save the downloaded files.')
    parser.add_argument('--overwrite', action='store_true', help='Flag to overwrite directory.')
    parser.add_argument('--not-path-only', action='store_true',
                        help='print additional information on the collected path.')
    parser.add_argument('--not-only-unique-paths', action='store_true',
                        help='output repeating paths.')
    parser.add_argument('--not-collapse-prepending-asns', action='store_true',
                        help='ouput prepending asns in paths (AS1 AS2 AS2 AS3).')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose output to stderr.')
    args = parser.parse_args()

    if os.path.isdir(args.directory) and not args.overwrite:
        print(f'[-] Directory {args.directory} already exists. Use --overwrite to overwrite.',
              file=sys.stderr)
        sys.exit()
    try:
        os.mkdir(args.directory)
    except FileExistsError:
        print(f'[*] Overwriting directory {args.directory}.', file=sys.stderr)

    main(args)
