from sklearn.model_selection import train_test_split
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)

    df_train = df_train.reset_index(drop=True)
    return df_train

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'