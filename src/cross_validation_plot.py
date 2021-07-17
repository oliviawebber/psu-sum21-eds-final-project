from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import load
import pandas as pd
import numpy as np

# List of all the  files we need to train models for
fileList = ['cleveland', 'hungarian', 'long-beach-va', 'switzerland']

# Setup an array for storing the accuracy results
results = np.zeros((4,4))

modelIndex = 0
for model in fileList:
    # Load the trained model
    rfClassifier = load('../models/{}-trained-model.joblib'.format(model))

    # Compute accuracy on each file
    fileIndex = 0
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

        
        # Save accuracy
        results[modelIndex][fileIndex] = rfClassifier.score(testData, testTargets)
        fileIndex += 1
    modelIndex += 1

# Round the results to 2 digits for easier printing
results = np.around(results, 2)

# Modify the file names into better labels
rowLabels = ['test-' + x for x in fileList]
colLabels = ['train-' + x for x in fileList]

# Setup and show the table
colors = plt.cm.hot(results)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,frameon=True, xticks=[], yticks=[])
table = plt.table(cellText=results, rowLabels=rowLabels, colLabels=colLabels, cellColours=colors)
plt.show()

