import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""
dataset = 'https://github.com/Hokfu/campaign_success_prediction_with_mlops/raw/master/campaign_data.parquet'
df = pd.read_parquet(dataset)
df.columns = df.columns.str.replace(' ','_').str.lower()
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in string_columns:
    df[c] = df[c].str.lower()
df = df.set_index('campaignid')

reference_data, current_data = train_test_split(df, test_size=0.5, random_state=42)
reference_data = reference_data.reset_index(drop=True)
current_data = current_data.reset_index(drop=True)

with open('models/rf_clf.bin', 'rb') as f_in:
    model = joblib.load(f_in)
    
with open('models/dv.bin', 'rb') as f_in:
    dv = joblib.load(f_in)

cat_features = list(df.dtypes[df.dtypes == 'object'].index)
num_features = list(df.dtypes[(df.dtypes == 'float64') | (df.dtypes == 'int64')].index)

reference_data_dict = reference_data[num_features + cat_features].to_dict(orient = 'records')
X_reference_data = dv.transform(reference_data_dict)
reference_data['prediction'] = model.predict(X_reference_data)


column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)
 

def calculate_metrics_postgresql(curr, i):
    current_data_dict = current_data[num_features + cat_features].to_dict(orient = 'records')
    X_current_data = dv.transform(current_data_dict)
    current_data['prediction'] = model.predict(X_current_data)

    report.run(reference_data = reference_data, current_data = current_data,
    column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    curr.execute(
    "insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
    (datetime.datetime.now(pytz.timezone('Asia/Bangkok')), prediction_drift, num_drifted_columns, share_missing_values)
    )
 
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 27):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
 batch_monitoring_backfill()