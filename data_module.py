import joblib
import pandas as pd

def export_train_test(train, test, Categorical):
    """Exports the training and testing dataframes to CSV files.

    Args:
        train (pd.DataFrame): The training dataframe.
        test (pd.DataFrame): The testing dataframe.
        Categorical (pd.Series): A series containing the names of categorical features.
    """
    print('> Saving data...')
    train.to_csv('./data/train_fe.csv', index=False)
    test.to_csv('./data/test_fe.csv', index=False)
    Categorical.to_csv('./data/categorical.csv')
    print('> Data saved.\n')


def import_train_test():
    """Imports the training and testing dataframes from CSV files.

    Returns:
        tuple: A tuple containing the training dataframe, testing dataframe, and a series of categorical feature names.
    """
    print('> Importing data...')
    train = pd.read_csv('./data/train_fe.csv')
    test = pd.read_csv('./data/test_fe.csv')
    Categorical = pd.read_csv('./data/categorical.csv')['0']

    # 3) Transform categorical
    train[Categorical] = train[Categorical].fillna('nan').astype(str)
    test[Categorical] = test[Categorical].fillna('nan').astype(str)

    print('> Data imported.\n')
    return train, test, Categorical


def export_model(model, name='./data/model_catboost.sav'):
    """Exports the trained model to a file using joblib.

    Args:
        model: The trained model to be saved.
        name (str, optional): The file path to save the model to. Defaults to './data/model_catboost.sav'.
    """
    joblib.dump(model, name)


def import_model(name = './data/model_catboost.sav'):
    """Imports a trained model from a file using joblib.

    Args:
        name (str, optional): The file path to load the model from. Defaults to './data/model_catboost.sav'.

    Returns:
        The loaded model.
    """
    return joblib.load(name)


