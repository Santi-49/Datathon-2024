import warnings
import numpy as np
import pandas as pd  # Install version 1.5.3 (iteritems errors)
import os

warnings.simplefilter(action='ignore', category=FutureWarning)  # Remove warning iteritems in Pool

# read data
print(os.listdir("./data"))
train = pd.read_csv('./data/train.csv').iloc[:5000] # Seleccionar primeras 5000 columnas
test = pd.read_csv('./data/test.csv')

