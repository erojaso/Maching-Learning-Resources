# load the minimum temperatures dataset
from matplotlib import pyplot
from pandas import read_csv
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
print(series.head())
series.plot()
pyplot.show()