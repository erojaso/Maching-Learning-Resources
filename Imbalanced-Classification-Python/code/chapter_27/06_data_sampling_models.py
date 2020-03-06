# compare data sampling with logistic regression on the oil spill dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	data = read_csv(full_path, header=None)
	# drop unused columns
	data.drop(22, axis=1, inplace=True)
	data.drop(0, axis=1, inplace=True)
	# retrieve numpy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	return X, y

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation the metric
	metric = make_scorer(geometric_mean_score)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define models to test
def get_models():
	models, names = list(), list()
	# SMOTEENN
	sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
	model = LogisticRegression(solver='liblinear')
	steps = [('e', sampling), ('m', model)]
	models.append(Pipeline(steps=steps))
	names.append('LR')
	# SMOTEENN + Norm
	sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
	model = LogisticRegression(solver='liblinear')
	steps = [('t', MinMaxScaler()), ('e', sampling), ('m', model)]
	models.append(Pipeline(steps=steps))
	names.append('Norm')
	# SMOTEENN + Std
	sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
	model = LogisticRegression(solver='liblinear')
	steps = [('t', StandardScaler()), ('e', sampling), ('m', model)]
	models.append(Pipeline(steps=steps))
	names.append('Std')
	# SMOTEENN + Power
	sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
	model = LogisticRegression(solver='liblinear')
	steps = [('t1', MinMaxScaler()), ('t2', PowerTransformer()), ('e', sampling), ('m', model)]
	models.append(Pipeline(steps=steps))
	names.append('Power')
	return models, names

# define the location of the dataset
full_path = 'oil-spill.csv'
# load the dataset
X, y = load_dataset(full_path)
# define models
models, names = get_models()
# evaluate each model
results = list()
for i in range(len(models)):
	# evaluate the model and store results
	scores = evaluate_model(X, y, models[i])
	# summarize and store
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
	results.append(scores)
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
