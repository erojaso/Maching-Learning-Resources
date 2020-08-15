# load the female births dataset
from matplotlib import pyplot
from pandas import read_csv
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
print(series.head())
series.plot()
pyplot.show()