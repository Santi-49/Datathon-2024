import joblib
import pandas as pd

def export_train_test(train, test, Categorical):
    print('> Saving data...')
    train.to_csv('./data/train_fe.csv', index=False)
    test.to_csv('./data/test_fe.csv', index=False)
    Categorical.to_csv('./data/categorical.csv')
    print('> Data saved.\n')


def import_train_test():
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
    joblib.dump(model, name)


def import_model(name = './data/model_catboost.sav'):
    return joblib.load(name)


