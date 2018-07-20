import numpy as np
from numpy.matlib import repmat
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import kstwobign
from scipy.stats import multivariate_normal as mvn
from qpoases import PyQProblem as QProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue as ReturnValue
from time import time
import json
import gc

def GaussKernel(X, Y=None):
	gamma = 2.5 / X.shape[1]
	if Y is None:
		pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
	else:
		pairwise_sq_dists = cdist(Y, X, 'sqeuclidean')
	return np.exp(-gamma*pairwise_sq_dists)

class QOCSVM():
	def __init__(self, alphas, X=None, param_path=None, kernel='gauss', gamma='auto', max_iter=1000, tol=1e-10, verbose=True):
		self.alphas = alphas
		self.kernel = GaussKernel if kernel=='gauss' else kernel
		self.gamma = gamma
		self.max_iter = max_iter
		self.etastars = None
		self.rhos = None
		self.X = X
		self.tol = tol
		self.verbose = verbose
		if param_path is not None:
			with open(param_path, 'r') as fp:
				params = json.load(fp)
				self.etastars = params['etastars']
				self.rhos = params['rhos']
	def fit(self, X, save_results=None):
		q, n = len(self.alphas), X.shape[0]
		self.X = X
		## formulate QP
		try:
			H = self.kernel(X)
		except:
			H = self.kernel(X,X)
		H = 1/2 * (H + np.transpose(H)) ## ensure symmetry
		H = repmat(H, q, q)
		g = np.zeros(q*n) ## no linear term
		A = np.zeros((q, q*n))
		lbA = np.ones(q) ## equality, thus ubA = lbA
		lb = np.zeros(q*n)
		ub = np.ones(q*n)

		for i in range(q):
			start = i*n + 1
			end = start + n -1
			ub[start:end] = 1 / (n*(1-self.alphas[i]))
			A[i, start:end] = 1

		## solve QP
		qp = QProblem(q*n, q)
		if not self.verbose:
			options = Options()
			options.printLevel = PrintLevel.NONE
			qp.setOptions(options)
		suc = qp.init(H, g, A, lb, ub, lbA, lbA, self.max_iter)
		if suc == ReturnValue.MAX_NWSR_REACHED:
			msg = "qPOASES reached the maximum number of iterations ({}). ".format(self.max_iter)
			msg += "\nThe resulting regions may not be reliable"
			print(msg)
		del A
		gc.collect()
		etas = np.zeros(q*n)
		qp.getPrimalSolution(etas);
		etas = etas.reshape(q,n)
		etastars = etas.sum(axis=0)
		nus = 1-self.alphas
		SVidx = np.arange(len(etastars))[etastars>self.tol]
		nSV = len(SVidx)
		ub = 1/n*nus
		rhos = np.zeros(q)
		for j, eta in enumerate(etas):
			choose = np.logical_and(eta > self.tol, eta < ub[j])
			hyperSVidx = np.arange(len(eta))[choose]
			#print(len(hyperSVidx))
			if len(hyperSVidx)==0:
				hyperSVidx = np.arange(len(eta))[eta > self.tol]
				rhos[j] = max(np.dot(H[hyperSVidx][:,SVidx], etastars[SVidx])/q)
			else:
				rhos[j] = np.median(np.dot(H[hyperSVidx][:,SVidx], etastars[SVidx])/q)
		self.rhos = rhos
		self.etastars = etastars
		if save_results is not None:
			out = dict(rhos=rhos, etastars=etastars)
			pd.Series(out).to_json(save_results, orient='index')
		return self
	def predict(self, X):
		if self.X is None or self.etastars is None or self.rhos is None:
			return None
		q = len(self.rhos)
		K = self.kernel(self.X, X)
		objFun = np.dot(K, self.etastars)/q
		out = [(objFun>rho+self.tol).astype(int) for rho in self.rhos]
		out = np.transpose(np.asarray(out))
		return out

'''
import pandas as pd


rv = mvn(mean=np.zeros(20), cov=1)
rv2 = mvn(mean=np.zeros(20), cov=.95)
X1 = rv.rvs(size=5000)
X2 = rv2.rvs(size=5000)
alphas = np.arange(1,6)/6

np.random.seed(666)

qocsvm = QOCSVM(alphas, X=X1, max_iter=20000, verbose=True, param_path=None)
qocsvm.fit(X1, save_results=None)
result = qocsvm.predict(X2)
result2 = qocsvm.predict(X1)

print(pd.DataFrame(result).mean(axis=1))
print(pd.DataFrame(result2).mean(axis=1))

print(pd.DataFrame(result).mean(axis=0))
print(pd.DataFrame(result2).mean(axis=0))
'''


