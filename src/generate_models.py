from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump
import pandas as pd
import numpy as np
import pathlib

# Setup the search grid for doing hyperparameter tuning
# Number of trees in the random forest
numTrees = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
maxFeatures = ['auto', 'sqrt']
# Maximum number of levels in tree
maxDepth = [int(x) for x in np.linspace(10, 110, num = 11)]
maxDepth.append(None)
# Minimum number of samples required to split a node
minSamplesSplit = [2, 5, 10]
# Minimum number of samples required at each leaf node
minSamplesLeaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
randomGrid = {
    'n_estimators': numTrees,
    'max_features': maxFeatures,
    'max_depth': maxDepth,
    'min_samples_split': minSamplesSplit,
    'min_samples_leaf': minSamplesLeaf,
    'bootstrap': bootstrap
}

# List of all the  files we need to train models for
fileList = ['cleveland', 'hungarian', 'long-beach-va', 'switzerland']

# Project root so we know where to save models
projectRoot = str(pathlib.Path(__file__).parent.parent.resolve())

for file in fileList:
    # Read in the clean data
    data = pd.read_csv(projectRoot + '/data/{}-clean.csv'.format(file), sep=',', header=0)

    # Remove the targets from the data and store them. Then convert the data
    # into a features matrix
    targets = np.array(data['num'])
    data = data.drop('num', axis=1)
    data = np.array(data)

    # Split the data into training and test sets
    trainData, testData, trainTargets, testTargets = train_test_split(data, targets, test_size=0.10, random_state=0)

    # Setup the model to be tuned and randomly tune it using cross validation
    tuningRandomForest = RandomForestClassifier()
    tunedRandomForest = RandomizedSearchCV(
        estimator = tuningRandomForest,
        param_distributions = randomGrid,
        n_iter = 1000,
        random_state=0,
        n_jobs = -1).fit(trainData, trainTargets)

    # Setup a new classifier with the tuned parameters
    modelParameters = tunedRandomForest.best_params_
    rfClassifier = RandomForestClassifier(
        n_estimators = modelParameters['n_estimators'],
        min_samples_split = modelParameters['min_samples_split'],
        min_samples_leaf = modelParameters['min_samples_leaf'],
        max_features = modelParameters['max_features'],
        max_depth = modelParameters['max_depth'],
        bootstrap = modelParameters['bootstrap'],
        random_state = 0,
    ).fit(trainData, trainTargets)

    # Save the newly trained model
    dump(rfClassifier, projectRoot + '/models/{}-trained-model.joblib'.format(file))
        
    
    
