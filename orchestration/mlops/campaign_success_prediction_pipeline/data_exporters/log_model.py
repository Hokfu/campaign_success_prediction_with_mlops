from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

import mlflow
import mlflow.sklearn
import pickle
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

if not os.path.exists('models'):
    os.mkdir('models')
else:
    print("Directory 'models' already exists.")
    
@data_exporter
def export_data(model_dv):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("campaign-success-prediction")
    rf_clf, dv = model_dv
    with open('models/rf_clf.bin','wb') as fout:
        pickle.dump(rf_clf,fout)
    with open('models/dv.bin','wb') as fout:
        pickle.dump(dv,fout)
    with mlflow.start_run():
        mlflow.log_artifact(local_path='models/dv.bin', artifact_path='dv_pickle')
        mlflow.sklearn.log_model(sk_model=rf_clf,
        artifact_path='models/rf_clf.bin',registered_model_name='random forest model')