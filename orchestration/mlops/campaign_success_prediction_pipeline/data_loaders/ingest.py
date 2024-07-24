import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    dataset = 'https://github.com/Hokfu/campaign_success_prediction_with_mlops/raw/master/campaign_data.parquet'
    df = pd.read_parquet(dataset)
    df.columns = df.columns.str.replace(' ','_').str.lower()
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in string_columns:
        df[c] = df[c].str.lower()
    df = df.set_index('campaignid')
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'