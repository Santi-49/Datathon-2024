import warnings
import numpy as np
import pandas as pd  # Install version 1.5.3 (iteritems errors)
import os

warnings.simplefilter(action='ignore', category=FutureWarning)  # Remove warning iteritems in Pool

# Read data
train = pd.read_csv('./data/train.csv')#.iloc[:5000] # Seleccionar primeras 5000 columnas
test = pd.read_csv('./data/test.csv')

pd.options.mode.chained_assignment = None
pd.set_option("display.max_rows", 50, "display.max_columns", None)


################################################################################
################################# FEATURE ######################################
############################### ENGINEERING ####################################
################################################################################
TARGET = 'Listing.Price.ClosePrice'
ID = 'Listing.ListingId'
Features = train.dtypes.reset_index()
Categorical = Features.loc[Features[0] == 'object', 'index'].drop(17)


# For numerical mode only
'''
train = train.drop(Categorical, axis=1)
test = test.drop(Categorical, axis=1)
Categorical = []
'''

# Move ID to front and TARGET to the back
columns = [ID] + [col for col in train.columns if col not in [ID, TARGET]] + [TARGET]
train = train[columns]
columns.pop()
test = test[columns]

# 0) Drop outliers
train = train[train[TARGET] < 13000000]


# 1) Drop irrelevant columns
# Function to print in every categorical column the number of unique values contained in it
def unique(ds):
    rows = len(ds)
    for col in Categorical:
        if len(ds[col].unique()) / rows > 0.50:
            print(col, len(ds[col].unique()) / rows) # Unique values / rows as a percentage

print('\n################## TRAIN ##################')
unique(train)
print('\n################## TEST ##################')
unique(test)

Columns_to_drop = ['Location.Address.StreetDirectionPrefix',
     'Location.Address.StreetDirectionSuffix',
    'Location.Address.StreetNumber', 'Location.Address.StreetSuffix',
     'Location.Address.UnitNumber', 'Location.Address.UnparsedAddress',
    'Listing.Dates.CloseDate', 'Location.Address.CensusBlock', 'Location.Address.PostalCodePlus4']
train = train.drop(columns=Columns_to_drop, axis=1)
test = test.drop(columns=Columns_to_drop, axis=1)


# 2) Unpack columns as lists
# ...
list_features = ["Characteristics.LotFeatures", "ImageData.features_reso.results", "ImageData.room_type_reso.results",
                "Structure.Basement", "Structure.Cooling", "Structure.Heating", "Structure.ParkingFeatures"]
def unpack_lists(train, test, list_features):
    added_cols = []
    for col in list_features:

        if col not in train.columns:
            print(f"Column '{col}' not found in train DataFrame.")
            continue  # Skip processing for missing columns

        # Expand list columns into one-hot encoded features
        new_features_df_train = expand_list_column(train, col)
        new_features_df_test = expand_list_column(test, col)

        #Threshold for filtering columns
        threshold = (train.shape[0] * 1) / 100

        # Filter train columns based on threshold
        columns_to_keep_train = new_features_df_train.columns[new_features_df_train.sum() > threshold]
        filtered_features_df_train = new_features_df_train[columns_to_keep_train]

        # Filter test by ensuring alignment with train columns
        columns_to_keep_test = coltest_in_coltrain(new_features_df_test.columns, columns_to_keep_train)
        filtered_features_df_test = new_features_df_test[columns_to_keep_test]
        added_cols.append(columns_to_keep_test)

        # Concatenate filtered features back into train and test DataFrames
        train = pd.concat([train, filtered_features_df_train], axis=1)
        train.drop(columns=[col], inplace=True)

        test = pd.concat([test, filtered_features_df_test], axis=1)
        test.drop(columns=[col], inplace=True)
    return (train, test, added_cols)


def coltest_in_coltrain(cols_test, cols_train):
    """
    Keep columns that both lists have in common.

    Returns:
    - list: A list with common columns.
    """
    common_cols = []
    for col in cols_test:
        if col in cols_train:
            common_cols.append(col)
    return common_cols


def expand_list_column(df, column):
    """
    Expands a column with list-like elements into multiple one-hot encoded columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The column name to process.

    Returns:
    - pd.DataFrame: A new DataFrame with one-hot encoded columns.
    """
    # Ensure the column has list-like elements
    expanded_data = df[column].dropna().apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Collect all elements into a flat list
    all_elements = [column+'.'+element for sublist in expanded_data for element in sublist]

    # Ensure unique elements
    unique_elements = pd.unique(all_elements)

    # Create a new DataFrame with one-hot encoding
    new_df = pd.DataFrame(0, index=df.index, columns=unique_elements)

    # Fill in the one-hot encoding
    for idx, values in expanded_data.items():
        for value in values:
            new_df.loc[idx, column+'.'+value] = 1

    return new_df

train, test, cols = unpack_lists(train, test, list_features)
added_cols = [col for feat in cols for col in feat]

columns = [ID] + [col for col in train.columns if col not in [ID, TARGET]] + [TARGET]
train = train[columns]


'''
train = train.drop(columns=list_features, axis=1)
test = test.drop(columns=list_features, axis=1)
'''

# 3) Change cat features
Features = train.dtypes.reset_index()
Categorical = Features.loc[(Features[0] == 'object'), 'index']
additional_features = added_cols.append('Location.Address.PostalCode')
Categorical = pd.concat([Categorical, pd.Series(additional_features)]).drop_duplicates()


# 3) Transform categorical
train[Categorical] = train[Categorical].fillna('nan').astype(str)
test[Categorical] = test[Categorical].fillna('nan').astype(str)

train.to_csv('./data/train_fe.csv', index=False)
test.to_csv('./data/test_fe.csv', index=False)
Categorical.to_csv('./data/categorical.csv')


################################################################################
################################ MODEL CATBOOST ################################
################################# TRAIN / TEST #################################
################################################################################
pred = list(train)[1:-1]
X_train = train[pred].reset_index(drop=True)
Y_train = train[TARGET].reset_index(drop=True)
X_test = test[pred].reset_index(drop=True)

# 1) For expensive models (catboost) we first try with validation set (no cv)
################################################################################
from catboost import CatBoostClassifier, CatBoostRegressor
from catboost import Pool

# train / test partition
RS = 124  # Seed for partitions (train/test) and model random part
TS = 0.3  # Validation size
esr = 100  # Early stopping rounds (when validation does not improve in these rounds, stops)

from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=TS, random_state=RS)

# Categorical positions for catboost
Pos = list()
As_Categorical = Categorical.tolist()
As_Categorical.remove(ID)
for col in As_Categorical:
    Pos.append((X_train.columns.get_loc(col)))


# To Pool Class (for catboost only)
pool_tr = Pool(x_tr, y_tr, cat_features=Pos)
pool_val = Pool(x_val, y_val, cat_features=Pos)


# By-hand paramter tuning. A grid-search is expensive
# We test different combinations
# See parameter options here:
# "https://catboost.ai/en/docs/references/training-parameters/"
model_catboost_val = CatBoostRegressor(
    loss_function='MAE',
    iterations=5000,  # Very high value, to find the optimum
    od_type='Iter',  # Overfitting detector set to "iterations" or number of trees
    od_wait=esr,
    random_seed=RS,  # Random seed for reproducibility
    verbose=100)  # Shows train/test metric every "verbose" trees


# "Technical" parameters of the model:
params = {'objective': 'MAE',
          'learning_rate': 0.1,  # learning rate, lower -> slower but better prediction
          # 'depth': 4,  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
          'min_data_in_leaf': 150,
          'l2_leaf_reg': 15,  # L2 regularization (between 3 and 20, higher -> less overfitting)
          'rsm': 0.5,  # % of features to consider in each split (lower -> faster and reduces overfitting)
          'subsample': 0.8,  # Sample rate for bagging
          'random_seed': RS}

'''

model_catboost_val.set_params(**params)

print('\nCatboost Fit (Validation)...\n')
model_catboost_val.fit(X=pool_tr,
                       eval_set=pool_val,
                       early_stopping_rounds=esr)
'''
nrounds = 3867
print('\nCatboost Optimal Fit with %d rounds...\n' % nrounds)
pool_train = Pool(X_train, Y_train, cat_features=Pos)
model_catboost = CatBoostRegressor(n_estimators=nrounds,
                                    random_seed=RS,
                                    verbose=100)
model_catboost.set_params(**params)

model_catboost.fit(X=pool_train)

from sklearn.externals import joblib
joblib.dump(model_catboost,'./data/model_catboost.sav')
model_catboost_uploaded = joblib.load('./data/model_catboost.sav')

################################################################################
################################# SHAP VALUES ##################################
################################################################################


import shap
import matplotlib.pyplot as plt


explainer = shap.TreeExplainer(model_catboost)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)

# Seleccionar una instancia específica del conjunto de prueba
sample_ind = 20  # Puedes cambiar este índice para seleccionar otra instancia
shap_values_test = explainer.shap_values(X_test)

# Convertir el shap_values_test a un objeto de tipo Explanation para la instancia seleccionada
shap_values_instance = shap_values_test[sample_ind]
shap_values_instance = shap.Explanation(values=shap_values_instance,
                                        base_values=explainer.expected_value,
                                        data=X_test.iloc[sample_ind])

# Gráfico de cascada para una instancia específica
shap.plots.waterfall(shap_values_instance, max_display=14)

################################################################################
################################### RESULTS ####################################
################################################################################

# Prediction (All train model)
test[TARGET] = model_catboost.predict(X_test)
catboost_submission = pd.DataFrame(test[[ID, TARGET]])

catboost_submission.to_csv('submission.csv', index=False)




