import numpy as np
from qocsvm import QOCSVM
from scipy.stats import kstwobign
from scipy.stats import multivariate_normal as mvn

np.random.seed(666)

def multi_ks2samp(X, Y, alphas, gamma=None):
	# two sample KS test in higher dimensions using QOCSVM sets
	m, n = X.shape[0], Y.shape[0]
	quants = QOCSVM(alphas, gamma=gamma)
	quants.fit(X)
	result1 = quants.transform(X)
	result2 = quants.transform(Y)

	F1 = result1.mean(axis=0)
	F2 = result2.mean(axis=0)
	max_delta = max(np.abs(F1-F2))
	teststat = np.sqrt((n * m) / (n + m)) * max_delta
	pval = kstwobign.sf(teststat)
	return pval, teststat, max_delta


rv1 = mvn(mean=np.zeros(10)).rvs(100)
rv2 = mvn(mean=np.zeros(10), cov=.8).rvs(700)

alphas = np.arange(1,10)/10
result = multi_ks2samp(rv1, rv2, alphas)
print(result)
