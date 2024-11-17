import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from main import X_train


def explainer(model_catboost):
    return shap.TreeExplainer(model_catboost)


def shap_summary(explainer, X_train):
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)
    return shap_values


def shap_explain(explainer, X_test, sample_ind):
    shap_values_test = explainer.shap_values(X_test)

    # Convertir el shap_values_test a un objeto de tipo Explanation para la instancia seleccionada
    shap_values_instance = shap_values_test[sample_ind]
    shap_values_instance = shap.Explanation(values=shap_values_instance,
                                            base_values=explainer.expected_value,
                                            data=X_test.iloc[sample_ind])

    # Gráfico de cascada para una instancia específica
    shap.plots.waterfall(shap_values_instance, max_display=14)
    return shap_values_test

TARGET = 'Listing.Price.ClosePrice'
ID = 'Listing.ListingId'

model_catboost_uploaded = joblib.load('./data/model_catboost.sav')
X_train = pd.read_csv('./data/X_train.csv')
Y_train = pd.read_csv('./data/Y_train.csv')
X_test = pd.read_csv('./data/X_test.csv')

Categorical = pd.read_csv('./data/categorical.csv')['0'].drop(0)
# 3) Transform categorical
X_train[Categorical] = X_train[Categorical].fillna('nann').astype(str)
X_test[Categorical] = X_test[Categorical].fillna('nann').astype(str)


exp = explainer(model_catboost_uploaded)
shap_values = shap_summary(exp, X_train)
shap_values_test = shap_explain(exp, X_test, sample_ind = 100)