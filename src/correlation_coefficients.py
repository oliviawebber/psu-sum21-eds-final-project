import sklearn as sk
import matplotlib.pyplot as plt
from joblib import load
import pandas as pd
import numpy as np
import pathlib

# List of all the  files we need to train models for
fileList = ['cleveland', 'hungarian', 'long-beach-va', 'switzerland']

# Setup an array for storing the accuracy results
results = np.zeros((4,4))

# Script variables
projectRoot = str(pathlib.Path(__file__).parent.parent.resolve())
for file in fileList:
    # Read in the clean data
    data = pd.read_csv(projectRoot + '/data/{}-clean.csv'.format(file), sep=',', header=0)

    # Remove the targets from the data and store them. Then convert the data
    # into a features matrix
    targets = np.array(data['num'])
    data = data.drop('num', axis=1)
    print(list(data.columns))
    data = np.array(data)
    corCoef = np.corrcoef(data, rowvar=False)
    np.savetxt(projectRoot + '/data/{}-coef.csv'.format(file), corCoef)

