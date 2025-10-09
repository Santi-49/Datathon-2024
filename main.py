import warnings
import numpy as np
import pandas as pd  # Install version 1.5.3
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
Categorical = Features.loc[Features[0] == 'object', 'index'].drop(17) #ID


# Move ID to front and TARGET to the back
columns = [ID] + [col for col in train.columns if col not in [ID, TARGET]] + [TARGET]
train = train[columns]
columns.pop()
test = test[columns]


######################## 0) Drop outliers ########################
train = train[train[TARGET] < 13000000]


#################### 1) Drop irrelevant columns ###################

# Function to print in every categorical column the number of unique values contained in it
def unique(ds):
    """Prints categorical columns with a high percentage of unique values.

    Args:
        ds (pd.DataFrame): The input dataframe.
    """
    rows = len(ds)
    print('Colums with unique values:\n')
    for col in Categorical:
        if len(ds[col].unique()) / rows > 0.50:
            print(col, len(ds[col].unique()) / rows) # Unique values / rows as a percentage


print('\n################## UNIQUENESS ##################')
unique(train)
print()

def missing_data(train, perc):
    """Displays columns with a high percentage of missing values.

    Args:
        train (pd.DataFrame): The input dataframe.
        perc (int): The percentage threshold for missing values.
    """
    # Calculate missing counts and percentages
    missing_data = train.isnull().sum()
    missing_percent = (missing_data / len(train)) * 100

    # Filter for columns with missing values
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percent': missing_percent
    }).sort_values(by='Missing_Count', ascending=False)

    # Filter for columns with any missing values
    missing_summary = missing_summary[missing_summary['Missing_Percent'] >= perc]

    # Display columns with missing values
    print("Columns with missing values:")
    print(missing_summary)


print('\n################## MISSING VAL ##################')
missing_data(train, 90)
print()


# Columns with irrelevant information
Columns_to_drop = ['Location.Address.StreetDirectionPrefix',
     'Location.Address.StreetDirectionSuffix',
    'Location.Address.StreetNumber', 'Location.Address.StreetSuffix',
     'Location.Address.UnitNumber', 'Location.Address.UnparsedAddress',
    'Listing.Dates.CloseDate', 'Location.Address.CensusBlock',
    'Location.Address.PostalCodePlus4', 'Location.Address.StateOrProvince']

train = train.drop(columns=Columns_to_drop, axis=1)
test = test.drop(columns=Columns_to_drop, axis=1)


############# 2) Unpack columns as lists (One hot encoding) #############
# Featoures that need unpacking (list format)
list_features = ["Characteristics.LotFeatures", "ImageData.features_reso.results", "ImageData.room_type_reso.results",
                "Structure.Basement", "Structure.Cooling", "Structure.Heating", "Structure.ParkingFeatures"]

def unpack_lists(train, test, list_features):
    """Unpacks columns containing lists into one-hot encoded features.

    Args:
        train (pd.DataFrame): The training dataframe.
        test (pd.DataFrame): The testing dataframe.
        list_features (list): A list of column names to unpack.

    Returns:
        tuple: A tuple containing the modified training and testing dataframes, and a list of the new column names.
    """
    added_cols = []
    for col in list_features:

        if col not in train.columns:
            print(f"Column '{col}' not found in train DataFrame.")
            continue  # Skip processing for missing columns

        # Expand list columns into one-hot encoded features
        new_features_df_train = expand_list_column(train, col)
        new_features_df_test = expand_list_column(test, col)

        #Threshold for filtering columns
        threshold = 0#(train.shape[0] * 1) / 100

        # Filter train columns based on threshold
        columns_to_keep_train = new_features_df_train.columns[new_features_df_train.sum() > threshold]
        filtered_features_df_train = new_features_df_train[columns_to_keep_train]

        # Filter test by ensuring alignment with train columns
        columns_to_keep_test, filtered_features_df_test = coltest_in_coltrain(new_features_df_test.columns, columns_to_keep_train, new_features_df_test)
        added_cols.append(columns_to_keep_test)

        # Concatenate filtered features back into train and test DataFrames
        train = pd.concat([train, filtered_features_df_train], axis=1)
        train.drop(columns=[col], inplace=True)

        test = pd.concat([test, filtered_features_df_test], axis=1)
        test.drop(columns=[col], inplace=True)
    return (train, test, added_cols)


def coltest_in_coltrain(cols_test, cols_train, features_test):
    """Aligns test columns with train columns, filling missing columns with zeros.

    Args:
        cols_test (list): List of test set columns.
        cols_train (list): List of train set columns to align with.
        features_test (pd.DataFrame): Test features DataFrame.

    Returns:
        tuple: A tuple containing a list of columns that overlap between train and test, and the aligned test DataFrame.
    """
    # Identify common and missing columns
    common_cols = [col for col in cols_test if col in cols_train]
    missing_cols = [col for col in cols_train if col not in cols_test]

    # Filter the test DataFrame for common columns
    aligned_test = features_test[common_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Add missing columns with zeros using .loc
    for col in missing_cols:
        aligned_test.loc[:, col] = 0

    # Ensure column order matches the train set
    aligned_test = aligned_test[cols_train]

    return common_cols, aligned_test


def expand_list_column(df, column):
    """Expands a column with list-like elements into multiple one-hot encoded columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to process.

    Returns:
        pd.DataFrame: A new DataFrame with one-hot encoded columns.
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


print('\n> Unpacking train and test data columns...')
train, test, cols = unpack_lists(train, test, list_features)
added_cols = [col for feat in cols for col in feat]
print('> Finished unpacking.\n')

# Move ID and TARGET
columns = [ID] + [col for col in train.columns if col not in [ID, TARGET]] + [TARGET]
train = train[columns]


################### 3) Change cat features ###################
Features = train.dtypes.reset_index()
Categorical = Features.loc[(Features[0] == 'object'), 'index'].drop(0)
added_cols.append('Location.Address.PostalCode')
Categorical = pd.concat([Categorical, pd.Series(added_cols)]).drop_duplicates()

# Transform categorical
train[Categorical] = train[Categorical].fillna('nan').astype(str)
test[Categorical] = test[Categorical].fillna('nan').astype(str)

############################ EXPORT DATA #######################################
from data_module import *

export_train_test(train, test, Categorical)

############################ IMPORT DATA #######################################
'''
TARGET = 'Listing.Price.ClosePrice'
ID = 'Listing.ListingId'
train, test, Categorical = import_train_test()
'''

################################################################################
################################ MODEL CATBOOST ################################
################################# TRAIN / TEST #################################
################################################################################
pred = list(train)[1:-1]
X_train = train[pred].reset_index(drop=True)
Y_train = train[TARGET].reset_index(drop=True)
X_test = test[pred].reset_index(drop=True)


# 1) For expensive models (catboost) we first try with validation set (no cv)

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
for col in As_Categorical:
    Pos.append((X_train.columns.get_loc(col)))


# To Pool Class (for catboost only)
pool_tr = Pool(x_tr, y_tr, cat_features=Pos)
pool_val = Pool(x_val, y_val, cat_features=Pos)


# By-hand paramter tuning
# We test different combinations
# Parameter options here:
# "https://catboost.ai/en/docs/references/training-parameters/"
model_catboost_val = CatBoostRegressor(
    loss_function='MAE',
    iterations=7000,  # Very high value, to find the optimum
    od_type='Iter',  # Overfitting detector set to "iterations" or number of trees
    od_wait=esr,
    random_seed=RS,  # Random seed for reproducibility
    verbose=100)  # Shows train/test metric every "verbose" trees


# "Technical" parameters of the model:
params = {'objective': 'MAE',
          'learning_rate': 0.05,  # learning rate, lower -> slower but better prediction
          # 'depth': 4,  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
          'min_data_in_leaf': 150,
          'l2_leaf_reg': 18,  # L2 regularization (between 3 and 20, higher -> less overfitting)
          'rsm': 0.4,  # % of features to consider in each split (lower -> faster and reduces overfitting)
          'subsample': 0.8,  # Sample rate for bagging
          'random_seed': RS}

'''
model_catboost_val.set_params(**params)

print('\nCatboost Fit (Validation)...\n')
model_catboost_val.fit(X=pool_tr,
                       eval_set=pool_val,
                       early_stopping_rounds=esr)
'''

############################ TRAIN WITH ALL DATA ###########################
nrounds = 7247
print('\nCatboost Optimal Fit with %d rounds...\n' % nrounds)
pool_train = Pool(X_train, Y_train, cat_features=Pos)
model_catboost = CatBoostRegressor(n_estimators=nrounds,
                                    random_seed=RS,
                                    verbose=100)
model_catboost.set_params(**params)

model_catboost.fit(X=pool_train)

############################### SAVE MODEL #####################################
from data_module import *

export_model(model_catboost, './data/model_catboost3.sav')

############################## IMPORT MODEL ####################################
'''
model_catboost = import_model()
model_catboost = import_model('./data/model_catboost3.sav')
'''
################################################################################
################################# SHAP VALUES ##################################
################################################################################

from shap_module import *

exp = explainer(model_catboost)
shap_values = shap_summary(exp, X_train)
shap_values_test = shap_explain(exp, X_test, 100)

################################################################################
################################### RESULTS ####################################
################################################################################

# Prediction (All train model)
test[TARGET] = model_catboost.predict(X_test)
catboost_submission = pd.DataFrame(test[[ID, TARGET]])

catboost_submission.to_csv('submission.csv', index=False)




