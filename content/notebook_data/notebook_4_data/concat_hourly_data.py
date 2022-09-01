import pandas as pd
import numpy as np

import os

files = os.listdir('.')
data_files = [e for e in files if e.startswith('en_climate_hourly')]

dfs = []
station_name = None
for f in data_files:
    df = pd.read_csv(
        f, 
        parse_dates=['Date/Time (LST)'], 
        infer_datetime_format=True, 
        index_col='Date/Time (LST)'
        )
    dfs.append(df)
    if station_name == None:
        station_name = df['Station Name'].values[0]
        stn_id = df['Climate ID'].values[0]
        fname = f'{station_name}_{stn_id}.csv'


all_data = pd.concat(dfs)
all_data.to_csv(fname)

for f in data_files:
    os.remove(f)
