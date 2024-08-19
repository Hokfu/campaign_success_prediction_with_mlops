import pandas as pd
import joblib

df = pd.read_csv('../campaign_data.csv')

# Preprocess data
df.columns = df.columns.str.replace(' ','_').str.lower()
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in string_columns:
    df[c] = df[c].str.lower()
df = df.set_index('campaignid')

# Load models
with open('../models/rf_clf.bin', 'rb') as f_in:
    model = joblib.load(f_in)
    
with open('../models/dv.bin', 'rb') as f_in:
    dv = joblib.load(f_in)

# Get feature lists
feature_columns = list(df.dtypes.index)[:-1]
feature_columns.remove('issuccessful')
# Prepare data for prediction
data_dict = df[feature_columns].to_dict(orient='records')
X_data = dv.transform(data_dict)
df['prediction'] = model.predict(X_data)

# Assert data preparation
assert df['prediction'].isnull().sum() == 0, "Predictions contain null values"
assert df['prediction'].dtype == 'int64', "Predictions are not integers"
assert df['prediction'].isin([0, 1]).all(), "Predictions are not binary"
