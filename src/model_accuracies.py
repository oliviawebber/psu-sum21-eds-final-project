from sklearn.model_selection import train_test_split
from joblib import load
import pandas as pd
import numpy as np

# List of all the  files we need to train models for
fileList = ['cleveland', 'hungarian', 'long-beach-va', 'switzerland']

for model in fileList:
    print('Model built on {}:'.format(model))
    # Load the trained model
    rfClassifier = load('../models/{}-trained-model.joblib'.format(model))

    # Compute accuracy on each file
    for file in fileList:
        # Read in the clean data
        data = pd.read_csv('../data/{}-clean.csv'.format(file), sep=',', header=0)

        # Remove the targets from the data and store them. Then convert the data
        # into a features matrix
        targets = np.array(data['num'])
        data = data.drop('num', axis=1)
        data = np.array(data)

        # Split the data into training and test sets
        trainData, testData, trainTargets, testTargets = train_test_split(data, targets, test_size=0.10, random_state=0)

        
        # Print accuracy
        print('Accuracy on file {}:'.format(file))
        print(rfClassifier.score(testData, testTargets))
    print('---')
