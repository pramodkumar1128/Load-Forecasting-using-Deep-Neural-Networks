import time
import pandas
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plot

# Measuring timestamp
print(int(time.time()))

# Load data from CSV files

# Load Training Data
CompleteTrainingData = pandas.read_csv('input2012and13.csv', header=0)
TrainingDataInput = CompleteTrainingData.iloc[0:17544, 1:6]  # Removing first column which is date
print('Training data')
print(TrainingDataInput.head())
TrainingDataInput = TrainingDataInput.values
TrainingOutput = CompleteTrainingData.iloc[0:17544, 6]
TrainingOutput = TrainingOutput.values

# Load Test Data
CompleteTestData = pandas.read_csv('input2014.csv', header=0)
TestDataInput = CompleteTestData.iloc[0:8760, 1:6]  # Removing first column which is date
print(TestDataInput.head())
TestDataInput = TestDataInput.values
TestDataOutput = CompleteTestData.iloc[0:8760, 6]
TestDataOutput = TestDataOutput.values

# Scaling the data
Scalar = StandardScaler()
TrainingDataInput = Scalar.fit_transform(TrainingDataInput)
TestDataInput = Scalar.transform(TestDataInput)
print(TestDataInput)

# Build Model

model = MLPRegressor(hidden_layer_sizes=100, activation='relu',
                     solver='lbfgs',
                     max_iter=10000, verbose=1)
model.fit(TrainingDataInput, TrainingOutput)

# Predict day
x = 61
start = 24 * x
end = 24 * (x + 1)
PredictingDay = model.predict(TestDataInput[start:end])

# Plotting results for one day
plot.plot(PredictingDay, color='blue')
plot.plot(TestDataOutput[start:end], color='red')
plot.ylabel('Test load 2014 for day')
plot.show()

# Printing MAPE
print('MAPE')
MapeDay = np.absolute((TestDataOutput[start: end] - PredictingDay) / TestDataOutput[start: end])
MapeDay = np.mean(np.absolute((TestDataOutput[start: end] - PredictingDay) / TestDataOutput[start: end])) * 100
print(MapeDay)
x0 = np.abs((TestDataOutput[start: end] - PredictingDay) / TestDataOutput[start: end]) * 100
plot.plot(x0)
plot.ylabel('MAPE error')
plot.show()

# Predict week
y = 5
start = (y * 24 * 7) + 1
end = (y + 1) * 24 * 7
PredictionsWeek = model.predict(TestDataInput[start: end])

# Plotting results for one week
plot.plot(PredictionsWeek, color='blue', label='Predicted')
plot.plot(TestDataOutput[start: end], color='red', label='Actual')
plot.ylabel('Test load 2014 for week')
plot.show()

# Printing MAPE
print('MAPE')
mape1 = np.mean(np.absolute((TestDataOutput[start: end] - PredictionsWeek) / TestDataOutput[start: end])) * 100
print(mape1)
x0 = np.abs((TestDataOutput[start: end] - PredictionsWeek) / TestDataOutput[start: end]) * 100
plot.plot(x0)
plot.ylabel('MAPE error')
plot.show()

# Predict year
PredictionsYear = model.predict(TestDataInput)

# Plotting results for one week
plot.plot(PredictionsYear, color='blue', label='Predicted')
plot.plot(TestDataOutput, color='red', label='Actual')
plot.ylabel('Test load 2014 year')
plot.show()

# Printing MAPE
print('MAPE')
mape1 = np.mean(np.absolute((TestDataOutput - PredictionsYear) / TestDataOutput)) * 100
print(mape1)
x0 = np.abs((TestDataOutput - PredictionsYear) / TestDataOutput) * 100
plot.plot(x0)
plot.ylabel('MAPE error')
plot.show()

# Measuring timestamp
print(int(time.time()))

