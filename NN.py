import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tensorflow.keras.optimizers import Adam

def predict_house_price_from_dataframes(train_df, test_df, id_col, lat_col, lon_col, target_col):
    # Eliminar filas con valores NaN en train_df y test_df
    train_df = train_df.dropna(subset=[lat_col, lon_col, target_col])
    test_df = test_df.dropna(subset=[lat_col, lon_col])

    # Separar características y el target
    X_train = train_df[[lat_col, lon_col]]
    y_train = train_df[target_col]
    X_test = test_df[[lat_col, lon_col]]

    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crear el modelo de la red neuronal
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,)),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1, activation='linear')
    ])

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.002), loss='mse', metrics=['mae'])

    # Entrenar el modelo
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

    # Predecir en el conjunto de prueba
    test_df['Prediction'] = model.predict(X_test_scaled)

    # Si el input es NaN, devolver NaN como predicción
    test_df.loc[test_df[[lat_col, lon_col]].isna().any(axis=1), 'Prediction'] = np.nan

    return model, test_df[[id_col, 'Prediction']]


ID = 'Listing.ListingId'
LAT = 'Location.GIS.Latitude'
LON = 'Location.GIS.Longitude'
TARGET = 'Listing.Price.ClosePrice'

train = pd.read_csv('./data/train.csv')
train = train[[ID, LON, LAT, TARGET]]
test = pd.read_csv('./data/test.csv')
test = test[[ID, LON, LAT]]

#pred = predict_house_price_from_dataframes(train, test, ID, LAT, LON, TARGET)

from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(train, test_size=0.3, random_state=42)

# Llamar a la función para entrenar y evaluar
model, test_predictions = predict_house_price_from_dataframes(
    train_split,
    val_split,  # Usamos la parte de validación como test en este caso
    id_col=ID,
    lat_col=LAT,
    lon_col=LON,
    target_col=TARGET
)


print(test_predictions.iloc[:10], val_split.iloc[:10])
print(sum(abs((test_predictions['Prediction'] - val_split[TARGET])).dropna())/len(abs((test_predictions['Prediction'] - val_split[TARGET])).dropna()))


def plot_price_heatmap(model, train_df, lat_col, lon_col, resolution=100):
    """
    Plots a heatmap of the predicted house prices based on latitude (y-axis) and longitude (x-axis).

    Parameters:
    - model: Trained model that predicts house prices.
    - train_df: DataFrame containing the training data to determine the plot's extent.
    - lat_col: Column name for latitude.
    - lon_col: Column name for longitude.
    - resolution: Number of points for the grid along each axis (default=100).
    """
    # Determine the range of latitudes and longitudes based on the training data
    lat_min, lat_max = train_df[lat_col].min(), train_df[lat_col].max()
    lon_min, lon_max = train_df[lon_col].min(), train_df[lon_col].max()

    # Create a grid of latitude and longitude values
    lat_grid, lon_grid = np.linspace(lat_min, lat_max, resolution), np.linspace(lon_min, lon_max, resolution)
    lat_lon_grid = np.array(np.meshgrid(lat_grid, lon_grid)).T.reshape(-1, 2)

    # Predict house prices for the grid points
    predictions = model.predict(lat_lon_grid)

    # Reshape predictions into a 2D grid for plotting
    price_grid = predictions.reshape(len(lon_grid), len(lat_grid))

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.contourf(lat_grid, lon_grid, price_grid.T, cmap='viridis', levels=50)  # Transpose grid for swapped axes
    plt.colorbar(label='Predicted Price')
    plt.scatter(train_df[lat_col], train_df[lon_col], color='red', s=10, label='Training Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('House Price Heat Map')
    plt.legend()
    plt.show()

plot_price_heatmap(
    model=model,
    train_df=train,
    lat_col='Location.GIS.Latitude',
    lon_col='Location.GIS.Longitude',
    resolution=2000  # Higher resolution for finer details
)

