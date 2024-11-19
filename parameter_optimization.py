from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import optuna
from data_module import *
from sklearn.model_selection import train_test_split
from datetime import datetime


# Define the objective function with data encapsulated
def objective_cla(trial, X_train, y_train, X_val, y_val):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 1, 10),
    }
    model = CatBoostClassifier(**param, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    return model.best_score_['validation']['Logloss']



# Define the objective function with data encapsulated
def objective_reg(trial, pool_tr, pool_val, esr, RS):
    model = CatBoostRegressor(
        loss_function='MAE',
        iterations=5500,  # Very high value, to find the optimum
        od_type='Iter',  # Overfitting detector set to "iterations" or number of trees
        od_wait=esr,
        random_seed=RS,  # Random seed for reproducibility
        verbose=400)  # Shows train/test metric every "verbose" trees

    param = {'objective': 'MAE',
              'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),  # learning rate, lower -> slower but better prediction
              'depth': trial.suggest_int('depth', 4, 10),  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
              'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 20),  # L2 regularization (between 3 and 20, higher -> less overfitting)
              'rsm': trial.suggest_float('rsm', 0.2, 0.8),  # % of features to consider in each split (lower -> faster and reduces overfitting)
              'subsample': trial.suggest_float('subsample', 0.5, 0.9),  # Sample rate for bagging
              'random_seed': RS}
    model.set_params(**param)
    print('\n\n\n--------------------------------\nfitting with params:', param)
    model.fit(X=pool_tr,
                           eval_set=pool_val,
                           early_stopping_rounds=esr)

    return model.best_score_['validation']['MAE']


def optimice_regresor(pool_tr, pool_val, esr, RS, n_trials):
    storage_path = f"sqlite:///optuna_study{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.db"
    
    study = optuna.create_study(direction='minimize',
                                storage=storage_path,
                                study_name="regressor_optimization",
                                load_if_exists=True)
    
    study.optimize(lambda trial: objective_reg(trial, pool_tr, pool_val, esr, RS), n_trials=n_trials)

    #study.optimize(lambda trial: objective_reg(trial, pool_tr, pool_val, esr, RS), n_trials=n_trials)

    print("Best parameters:", study.best_params)
    with open(f'Best parameters{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', "w") as file:
        file.write(f'{study.best_params}')
    return study.best_params


if __name__ == "__main__":
    TARGET = 'Listing.Price.ClosePrice'
    ID = 'Listing.ListingId'
    train, test, Categorical = import_train_test()

    pred = list(train)[1:-1]
    X_train = train[pred].reset_index(drop=True)
    Y_train = train[TARGET].reset_index(drop=True)
    X_test = test[pred].reset_index(drop=True)

    # train / test partition
    RS = 1234  # Seed for partitions (train/test) and model random part
    TS = 0.3  # Validation size
    esr = 100  # Early stopping rounds (when validation does not improve in these rounds, stops)

    x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=TS, random_state=RS)

    # Categorical positions for catboost
    Pos = list()
    As_Categorical = Categorical.tolist()
    for col in As_Categorical:
        Pos.append((X_train.columns.get_loc(col)))

    pool_tr = Pool(x_tr, y_tr, cat_features=Pos)
    pool_val = Pool(x_val, y_val, cat_features=Pos)

    optimice_regresor(pool_tr, pool_val, esr, RS, 50)

