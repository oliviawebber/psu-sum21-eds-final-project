import csv

datasets = ['cleveland', 'hungarian', 'long-beach-va', 'switzerland']
attributeIndexes = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50]
numAttributes = len(attributeIndexes)
classIndex = 57

for dataset in datasets:
    with open('../data/{}-clean.csv'.format(dataset), 'w', newline='') as w:
        with open('../data/{}.data'.format(dataset), 'r') as f:
            csvWriter = csv.writer(w, delimiter=',')
            dataPoint = list()
            for line in f:
                line = line.split(' ')
                line[-1] = line[-1].strip()
                if line[-1] == 'name':
                    filteredDataPoint = [0] * (numAttributes+1)
                    for n in range(numAttributes):
                        filteredDataPoint[n] = float(dataPoint[attributeIndexes[n]])
                        filteredDataPoint[-1] = float(dataPoint[classIndex])

                    csvWriter.writerow(filteredDataPoint)
                    dataPoint = list()
                else:
                    dataPoint.extend(line)
    
