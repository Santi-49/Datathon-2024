import warnings
import numpy as np
import pandas as pd  # Install version 1.5.3 (iteritems errors)
import os

warnings.simplefilter(action='ignore', category=FutureWarning)  # Remove warning iteritems in Pool

# read data
print(os.listdir("./data"))
train = pd.read_csv('./data/train.csv') #.iloc[:5000] # Seleccionar primeras 5000 columnas
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
Categorical = Features.loc[Features[0] == 'object', 'index']
Numerical = Features.loc[Features[0] != 'object', 'index']

# Move ID to front and TARGET to the back
columns = [ID] + [col for col in train.columns if col not in [ID, TARGET]] + [TARGET]
train = train[columns]
columns.pop()
test = test[columns]


# 1) Missings
################################################################################
# Function to print columns with at least n_miss missings
def miss(ds, n_miss):
    miss_list = list()
    for col in list(ds):
        if ds[col].isna().sum() >= n_miss:
            print(col, ds[col].isna().sum(), ds[col].isna().sum()/5000)
            miss_list.append(col)
    return miss_list

# Which columns have 1 missing at least...
print('\n################## TRAIN ##################')
m_tr = miss(train, 1)
print('\n################## TEST ##################')
m_te = miss(test, 1)


train = train.drop(Categorical, axis=1)
test = test.drop(Categorical, axis=1)
Categorical = []



# 2) Correlations
################################################################################
# Let's see if certain columns are correlated
# or even that are the same with a "shift"
thresholdCorrelation = 0.99

def InspectCorrelated(df):
    corrMatrix = df.corr().abs()  # Correlation Matrix
    upperMatrix = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))
    correlColumns = []
    for col in upperMatrix.columns:
        correls = upperMatrix.loc[upperMatrix[col] > thresholdCorrelation, col].keys()
        if len(correls) >= 1:
            correlColumns.append(col)
            print("\n", col, '->', end=" ")
            for i in correls:
                print(i, end=" ")
    print('\nSelected columns to drop:\n', correlColumns)
    return correlColumns, corrMatrix


# Look at correlations in the original features
correlColumns, corrMatrix = InspectCorrelated(train)

# If we are ok, throw them:
train = train.drop(correlColumns, axis=1)
test = test.drop(correlColumns, axis=1)

# 3) Constants
################################################################################
# Let's see if there is some constant column:
def InspectConstant(df):
    consColumns = []
    for col in list(df):
        if len(df[col].unique()) < 2:
            print(df[col].dtypes, '\t', col, len(df[col].unique()))
            consColumns.append(col)
    print('\nSelected columns to drop:\n', consColumns)
    return consColumns


consColumns = InspectConstant(train.iloc[:, len(Categorical):-1])

# If we are ok, throw them:
train = train.drop(consColumns, axis=1)
test = test.drop(consColumns, axis=1)

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
#As_Categorical = Categorical.tolist()
#As_Categorical.remove('ID')
#for col in As_Categorical:
#    Pos.append((X_train.columns.get_loc(col)))

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






