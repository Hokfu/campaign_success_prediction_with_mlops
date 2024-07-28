import pickle
import pandas as pd
import sys

with open('./models/rf_clf.bin', 'rb') as f:
    rf_clf = pickle.load(f)

with open('./models/dv.bin', 'rb') as f:
    dv = pickle.load(f)

def load_data(file_path: str) -> pd.DataFrame:
    dataset = file_path
    df = pd.read_parquet(dataset)
    df.columns = df.columns.str.replace(' ','_').str.lower()
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in string_columns:
        df[c] = df[c].str.lower()
    df = df.set_index('campaignid')
    return df

def predict_success(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = list(df.dtypes.index)[:-1]
    df_dict = df[feature_columns].to_dict(orient='records')
    X = dv.transform(df_dict)
    y_pred = rf_clf.predict(X)
    total_success = sum(y_pred)
    total_not_success = len(y_pred) - total_success
    return f'Total successful campaigns: {total_success}\nTotal unsuccessful campaigns: {total_not_success}'

def run():
    file_path = sys.argv[1]
    df = load_data(file_path=file_path)
    result = predict_success(df)
    print(result)

if __name__ == '__main__':
    run()