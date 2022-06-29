#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd

categorical = ['PUlocationID', 'DOlocationID']

def mean_predict_durations(year, month) -> float:
    filename = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(filename, year, month)
    predictions_df = predict(df)

    mean_pred = predictions_df['predicitions'].mean()
    print(mean_pred)
    
    return mean_pred



def read_data(filename, year, month):
    
    df = pd.read_parquet(filename)
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df


def predict(df: pd.DataFrame) -> pd.DataFrame:

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df_result = pd.DataFrame(df['ride_id'])
    df_result['predicitions'] = y_pred

    return df_result


if __name__ == '__main__':
    mean_predict_durations(2021, 3)