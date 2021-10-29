#!/usr/bin/env python3

import argparse
import datetime as dt
import os
import pandas as pd
import pybgpstream
import pytz
import sys

from datetime import timezone
from tqdm import tqdm

NDAYS_BEFORE_ANOMALY = 2
NDAYS_AFTER_ANOMALY = 2

REGULAR_EVENT_LABEL = -1


def get_collection_start_and_finish_str_datetime(anomaly_start_date, anomaly_finish_date):
    ds = anomaly_start_date - dt.timedelta(days=NDAYS_BEFORE_ANOMALY)
    df = anomaly_finish_date + dt.timedelta(days=NDAYS_AFTER_ANOMALY)

    return ds, df


def datetime_to_str(d):
    return d.strftime('%Y-%m-%d %H:%M:%S')


def str_to_datetime(str_d):
    return pytz.utc.localize(dt.datetime.fromisoformat(str_d))


def main(args):

    events_df = pd.read_csv(args.events_file)

    tqdm_total = len(events_df)

    print('Considering the following events:', file=sys.stderr)
    print(events_df, file=sys.stderr)

    for i, event_data in events_df.iterrows():

        event = event_data.event
        project = event_data.project
        collector = event_data.collector

        event_no_spaces = event.replace(' ', '-')
        fname = f'{event_no_spaces}.{project}.{collector}.updates'

        f = open(os.path.join(args.directory, fname), 'w')

        anom_start_date = str_to_datetime(event_data.start_date)
        anom_finish_date = str_to_datetime(event_data.finish_date)

        start_time, finish_time = get_collection_start_and_finish_str_datetime(anom_start_date,
                                                                               anom_finish_date)

        print(f'Collecting data from {project}/{collector} on event {event}', file=sys.stderr)
        print(f'Time interval: from {start_time} to {finish_time}')

        stream = pybgpstream.BGPStream(
            from_time=datetime_to_str(start_time), until_time=datetime_to_str(finish_time),
            project=project,
            collectors=[collector],
            record_type='updates',
        )

        total_seconds = int((finish_time - start_time).total_seconds())
        with tqdm(total=total_seconds) as progress_bar:
            nseconds = 0
            last_nseconds = 0
            for elem in stream:
                d = dt.datetime.fromtimestamp(elem.time, tz=timezone.utc)
                if anom_start_date <= d <= anom_finish_date:
                    label = event_data['class']
                else:
                    label = REGULAR_EVENT_LABEL

                f.write(str(elem) + f'|{label}\n')

                nseconds = int((d - start_time).total_seconds())
                if nseconds > last_nseconds:
                    progress_bar.update(nseconds - last_nseconds)
                    last_nseconds = nseconds

        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_file', help='CSV file with containing the information of the \
                         events to consider.')
    parser.add_argument('directory', help='Directory in which to save the downloaded files.')
    parser.add_argument('--overwrite', action='store_true', help='Flag to overwrite directory.')
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
