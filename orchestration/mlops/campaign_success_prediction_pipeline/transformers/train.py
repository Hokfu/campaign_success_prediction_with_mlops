from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df_train):
    y_train = df_train.issuccessful
    feature_columns = list(df_train.dtypes.index)[:-1]

    train_dict = df_train[feature_columns].to_dict(orient = 'records')
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(train_dict)
    rf_clf = RandomForestClassifier(n_estimators=110,
                                    max_depth=10,
                                    min_samples_leaf=3)
    rf_clf.fit(X_train, y_train)
    return rf_clf, dv


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'