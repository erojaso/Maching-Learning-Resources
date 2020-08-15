# reframe regression as classification
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
# load data
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t', 't+1']
# make discrete
for i in range(len(dataframe['t+1'])):
	value = dataframe['t+1'][i]
	if value < 10.0:
		dataframe['t+1'][i] = 0
	elif value >= 25.0:
		dataframe['t+1'][i] = 2
	else:
		dataframe['t+1'][i] = 1
print(dataframe.head(5))