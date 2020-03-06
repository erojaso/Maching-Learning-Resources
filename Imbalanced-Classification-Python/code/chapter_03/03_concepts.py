# vary the number of clusters for a 1:100 imbalanced dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
# number of clusters
clusters = [1, 2]
# create and plot a dataset with different numbers of clusters
for i in range(len(clusters)):
	c = clusters[i]
	# define dataset
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=c, weights=[0.99], flip_y=0, random_state=1)
	counter = Counter(y)
	# define subplot
	pyplot.subplot(1, 2, 1+i)
	pyplot.title('Clusters=%d' % c)
	pyplot.xticks([])
	pyplot.yticks([])
	# scatter plot of examples by class label
	for label, _ in counter.items():
		row_ix = where(y == label)[0]
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
	pyplot.legend()
# show the figure
pyplot.show()