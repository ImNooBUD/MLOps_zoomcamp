from calendar import month
from multiprocessing import log_to_stderr
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.orion.schemas.schedules import CronSchedule
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner

from dateutil.relativedelta import relativedelta
from datetime import datetime, date


import pickle

@task(name="read_data")
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task(name="prepare_paths")
def get_path(date: date = None, 
            train_shift = relativedelta(months = 2), 
            val_shift  = relativedelta(months = 1)
                ):
    
    logger = get_run_logger()
    if date is None:
        date = datetime.today()
    logger.info(f"Get date = {date}")
    train = (date - train_shift).strftime('%Y-%m')
    val = (date - val_shift).strftime('%Y-%m')
    paths = list()
    for year_month in [train, val]:
        paths.append(f"../data/fhv_tripdata_{year_month}.parquet")
        logger.info(f"Path to file: {paths[-1]}")
    
    return paths



@task(name="prepare_features")
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task(name="train_model")
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task(name="run_model")
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(name="main_flow")
def main(run_date: date = date(2021,3,15)):
    logger = get_run_logger()
    train_path, val_path = get_path(run_date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    date_str = run_date.strftime('%Y-%m-%d')
    pickle.dump(lr, open(f'./artifacts/model-{date_str}.bin','wb'))
    logger.info(f"Save fitted model to file: ./artifacts/model-{date_str}.bin")
    pickle.dump(dv, open(f'./artifacts/dictvect-{date_str}.bin','wb'))
    logger.info(f"Save fitted dictvect to file: ./artifacts/dictvect-{date_str}.bin")

    lr = pickle.load(open(f'./artifacts/model-{date_str}.bin','rb'))
    dv = pickle.load(open(f'./artifacts/dictvect-{date_str}.bin','rb'))
    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    name = "cron-model-training-deployment",
    flow = main,
    flow_runner = SubprocessFlowRunner(),
    schedule=CronSchedule(cron="0 9 15 * *"),
    tags=['ml_trip']
)


if __name__ == "__main__":
    main(date(2021,8,15))
