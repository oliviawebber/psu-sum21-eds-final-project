import csv

# List of all the datasets to make looping over them easier
datasets = ['cleveland', 'hungarian', 'long-beach-va', 'switzerland']

# Information about where the relevant attributes are stored, these
# 13 attributes + the class are what are commonly used
attributeIndexes = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50]
dataHeaders = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
numAttributes = len(attributeIndexes)
classIndex = 57

# Loop over every dataset, cleaning the data
for dataset in datasets:
    with open('../data/{}-clean.csv'.format(dataset), 'w', newline='') as w:
        with open('../data/{}.data'.format(dataset), 'r') as f:
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
                    # Select out the desired attributes and cast them to floats
                    filteredDataPoint = [0] * (numAttributes+1)
                    for n in range(numAttributes):
                        filteredDataPoint[n] = float(dataPoint[attributeIndexes[n]])
                        filteredDataPoint[-1] = float(dataPoint[classIndex])

                    # Save the data point
                    csvWriter.writerow(filteredDataPoint)
                    dataPoint = list()
                # If the last entry is not name, extend the data point and
                # continue on. This logic is necessary because the original
                # data set has data points split over multiple lines
                else:
                    dataPoint.extend(line)
    
