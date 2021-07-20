import csv
import pathlib

# Toggles between binary and default classes in the clean dataset
BINARY_CLASSES = True

# List of all the datasets to make looping over them easier
datasets = ['cleveland', 'hungarian', 'long-beach-va', 'switzerland']

# Information about where the relevant attributes are stored, these
# 13 attributes + the class are what are commonly used
attributeIndexes = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50]
dataHeaders = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
numAttributes = len(attributeIndexes)
classIndex = 57

# Fetch the project root string so we know where to read and save to
projectRoot = str(pathlib.Path(__file__).parent.parent.resolve())

# Loop over every dataset, cleaning the data
for dataset in datasets:
    with open(projectRoot + '/data/{}-clean.csv'.format(dataset), 'w', newline='') as w:
        with open(projectRoot + '/data/{}.data'.format(dataset), 'r') as f:
            csvWriter = csv.writer(w, delimiter=',') 
            dataPoint = list()

            # Write the header line
            csvWriter.writerow(dataHeaders)
            for line in f:
                # For each line, it is space seperated so split the entries
                # and remove the \n character from the last entry
                line = line.split(' ')
                line[-1] = line[-1].strip()

                # If the last entry is name, a full data point has been
                # processed so select the relevant attributes and save it
                if line[-1] == 'name':
                    # Select out the desired attributes and cast them to floats, the class
                    # should also be downcasted to a binary variable
                    # i.e. 0 => 0 No presence 1,2,3,4 => 1 Presence
                    # The data set is not large enough to support a more detailed classification
                    # scheme
                    filteredDataPoint = [0] * (numAttributes+1)
                    for n in range(numAttributes):
                        filteredDataPoint[n] = float(dataPoint[attributeIndexes[n]])
                        dataPointClass = float(dataPoint[classIndex])
                        
                        if BINARY_CLASSES and dataPointClass > 0.0:
                            dataPointClass = 1.0
                        filteredDataPoint[-1] = dataPointClass

                    # Save the data point
                    csvWriter.writerow(filteredDataPoint)
                    dataPoint = list()
                # If the last entry is not name, extend the data point and
                # continue on. This logic is necessary because the original
                # data set has data points split over multiple lines
                else:
                    dataPoint.extend(line)
    
