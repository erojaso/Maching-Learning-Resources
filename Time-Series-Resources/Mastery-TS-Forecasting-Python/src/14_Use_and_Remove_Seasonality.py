# https://code.visualstudio.com/docs/python/jupyter-support
# 
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Chapter 14 - Use and Remove Seasonality
# 

# %%
# deseasonalize a time series using differencing

from pandas import read_csv
from matplotlib import pyplot
from pylab import rcParams

csvfile = '.\\code\\chapter_14\\daily-minimum-temperatures.csv'

series = read_csv(csvfile, header=0, index_col=0, parse_dates=True, squeeze=True)

X = series.values
diff = list()

# We can subtract the daily minimum temperature from the same day last year to correct for seasonality.
days_in_year = 365

for i in range(days_in_year, len(X)):
    value = X[i] - X[i - days_in_year]
    diff.append(value)

rcParams['figure.figsize'] = 15, 5
pyplot.plot(diff)
pyplot.show()


# %%


