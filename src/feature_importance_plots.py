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
    # Load the trained model and labels
    rfClassifier = load('../models/{}-trained-model.joblib'.format(model))
    data = pd.read_csv('../data/cleveland-clean.csv', sep=',', header=0)
    labels = list(data.columns)[:-1]

    # Setup and display the plots
    labelMarks = np.arange(len(labels))
    fig, ax = plt.subplots()
    bars = ax.bar(labelMarks, rfClassifier.feature_importances_)

    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance for Model Trained on {}'.format(model))
    ax.set_xticks(labelMarks)
    ax.set_xticklabels(labels,rotation=45)
    
    plt.show()



