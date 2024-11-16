import warnings
import numpy as np
import pandas as pd  # Install version 1.5.3 (iteritems errors)
import os

warnings.simplefilter(action='ignore', category=FutureWarning)  # Remove warning iteritems in Pool

# Read data
train = pd.read_csv('./data/train.csv').iloc[:5000] # Seleccionar primeras 5000 columnas
test = pd.read_csv('./data/test.csv').iloc[:5000]

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
    'Listing.Dates.CloseDate', 'Location.Address.CensusBlock']
train = train.drop(columns=Columns_to_drop, axis=1)
test = test.drop(columns=Columns_to_drop, axis=1)


# 2) Unpack columns as lists
# ...
list_features = ["Characteristics.LotFeatures", "ImageData.features_reso.results", "ImageData.room_type_reso.results",
                 "Structure.Basement", "Structure.Cooling", "Structure.Heating", "Structure.ParkingFeatures"]
train = train.drop(columns=list_features, axis=1)
test = test.drop(columns=list_features, axis=1)

Features = train.dtypes.reset_index()
Categorical = Features.loc[Features[0] == 'object', 'index'].drop(17)

# 3) Transform categorical
train[Categorical] = train[Categorical].fillna('nan').astype(str)
test[Categorical] = test[Categorical].fillna('nan').astype(str)




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
RS = 1234  # Seed for partitions (train/test) and model random part
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
    loss_function='RMSE',
    iterations=5000,  # Very high value, to find the optimum
    od_type='Iter',  # Overfitting detector set to "iterations" or number of trees
    od_wait=esr,
    random_seed=RS,  # Random seed for reproducibility
    verbose=100)  # Shows train/test metric every "verbose" trees

# "Technical" parameters of the model:
params = {'objective': 'RMSE',
          'learning_rate': 0.1,  # learning rate, lower -> slower but better prediction
          # 'depth': 4,  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
          'min_data_in_leaf': 150,
          'l2_leaf_reg': 15,  # L2 regularization (between 3 and 20, higher -> less overfitting)
          'rsm': 0.5,  # % of features to consider in each split (lower -> faster and reduces overfitting)
          'subsample': 0.8,  # Sample rate for bagging
          'random_seed': RS}

model_catboost_val.set_params(**params)

print('\nCatboost Fit (Validation)...\n')
model_catboost_val.fit(X=pool_tr,
                       eval_set=pool_val,
                       early_stopping_rounds=esr)






