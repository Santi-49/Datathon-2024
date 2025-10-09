import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd


def explainer(model_catboost):
    """Creates a SHAP TreeExplainer for the given model.

    Args:
        model_catboost: The trained CatBoost model.

    Returns:
        shap.TreeExplainer: The SHAP explainer.
    """
    return shap.TreeExplainer(model_catboost)


def shap_summary(explainer, X_train):
    """Generates and plots a SHAP summary plot.

    Args:
        explainer (shap.TreeExplainer): The SHAP explainer.
        X_train (pd.DataFrame): The training data.

    Returns:
        np.ndarray: The SHAP values.
    """
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)
    return shap_values


def shap_explain(explainer, X_test, sample_ind):
    """Generates and plots a SHAP waterfall plot for a specific instance.

    Args:
        explainer (shap.TreeExplainer): The SHAP explainer.
        X_test (pd.DataFrame): The test data.
        sample_ind (int): The index of the instance to explain.

    Returns:
        np.ndarray: The SHAP values for the test data.
    """
    shap_values_test = explainer.shap_values(X_test)

    # Convertir el shap_values_test a un objeto de tipo Explanation para la instancia seleccionada
    shap_values_instance = shap_values_test[sample_ind]
    shap_values_instance = shap.Explanation(values=shap_values_instance,
                                            base_values=explainer.expected_value,
                                            data=X_test.iloc[sample_ind])

    # Gráfico de cascada para una instancia específica
    plt.figure(figsize=(12, 8)) 
    
    # Ajustar la amplitud del eje X
    plt.xlim(left=shap_values_instance.base_values - 1.5 * abs(shap_values_instance.values).max(),
             right=shap_values_instance.base_values + 1.5 * abs(shap_values_instance.values).max())
    
    # Mejorar la claridad del texto en el eje Y
    plt.yticks(fontsize=12)  # Ajusta el tamaño de la fuente del eje Y
    plt.xticks(fontsize=12)  # Ajusta el tamaño de la fuente del eje X
    plt.tight_layout()  # Ajusta el diseño para evitar solapamientos
    shap.plots.waterfall(shap_values_instance, max_display=20)
    
    return shap_values_test


def main():
    """Main function to load data, model and generate SHAP plots."""
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
    #shap_values_test = shap_explain(exp, X_test, 100)

if __name__ == '__main__':
    main()

