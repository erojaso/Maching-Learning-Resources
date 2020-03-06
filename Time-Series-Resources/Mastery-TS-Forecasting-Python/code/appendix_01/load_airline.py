# load the airline passengers dataset
from matplotlib import pyplot
from pandas import read_csv
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
print(type(series))
print(series.head())
series.plot()
pyplot.show()