from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
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
modelIndex = 0
projectRoot = str(pathlib.Path(__file__).parent.parent.resolve())
for model in fileList:
    # Load the trained model and labels
    data = pd.read_csv(projectRoot + '/data/cleveland-clean.csv', sep=',', header=0)
    labels = list(data.columns)[:-1]

    # Remove the targets from the data and store them. Then convert the data
    # into a features matrix
    targets = np.array(data['num'])
    data = data.drop('num', axis=1)
    data = np.array(data)

    # Split the data into training and test sets
    trainData, testData, trainTargets, testTargets = trainData, testData, trainTargets, testTargets = train_test_split(
        data, targets, test_size=0.10, random_state=0)
    gauss = GaussianNB()
    gauss.fit(trainData, trainTargets)
    imps = permutation_importance(gauss, trainData, trainTargets)
    importances = imps.importances_mean
    std = imps.importances_std


    # Setup and display the plots
    labelMarks = np.arange(len(labels))
    fig, ax = plt.subplots()
    bars = ax.bar(labelMarks, importances)

    ax.set_ylabel('Importance')
    ax.set_title('Permutation test on {}'.format(model))
    ax.set_xticks(labelMarks)
    ax.set_xticklabels(labels,rotation=45)
    
    plt.show()



       
