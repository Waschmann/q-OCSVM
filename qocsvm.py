import numpy as np
from numpy.matlib import repmat
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import kstwobign
from qpoases import PyQProblem as QProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue as ReturnValue

def GaussKernel(X, Y=None, gamma=None):
	if gamma is None:
		gamma = 1 / X.shape[1]
	if Y is None:
		pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
	else:
		pairwise_sq_dists = cdist(Y, X, 'sqeuclidean')
	return np.exp(-gamma*pairwise_sq_dists)

class QOCSVM():
	'''
	Implementation of q-OCSVM, a method for estimating q nested 
	minimum-volume quantile regions of high dimensional distributions.

	Paper: https://papers.nips.cc/paper/5125-q-ocsvm-a-q-quantile-estimator-for-high-dimensional-distributions

	Parameters:
		alphas: 	Array of values 0 < alpha_i < 1
		kernel: 	String or function(X, Y), which computes the kernel matrix.
					Available options: 'gauss'
		gamma: 		Parameter of the gaussian kernel; Default (None) uses 1/dimension
		solver:		Currently unused
		max_iter: 	Maximum number of iterations for the QP solver. 
		tol: 		Tolerance
		verbose:	Verbosity of the QP solver

	Methods:
		fit: 		Computes the q=length(alphas) decision functions, given a set of samples X1
		transform:	Applies the decision functions on a set of samples X2
	'''
	def __init__(self, alphas, kernel='gauss', gamma=None, solver='qPOASES', max_iter=5000, tol=1e-10, verbose=False):
		self.alphas = np.sort(alphas)
		self.kernel = kernel if isinstance(kernel, str) else None
		self.kernel_fun = GaussKernel if kernel=='gauss' else kernel
		self.gamma = gamma
		self.max_iter = max_iter
		self.etastars = None
		self.rhos = None
		self.rhobounds = None
		self.X = None
		self.tol = tol
		self.verbose = verbose
	def fit(self, X):
		q, n = len(self.alphas), X.shape[0]
		self.X = X
		if self.kernel=='gauss':
			H = self.kernel_fun(X, gamma=self.gamma)
		else:
			H = self.kernel_fun(X, X)

		H = 1/2 * (H + np.transpose(H))
		H = repmat(H, q, q)
		g = np.zeros(q*n)
		A = np.zeros((q, q*n))
		lbA = np.ones(q)
		lb = np.zeros(q*n)
		ub = np.ones(q*n)
		for i in range(q):
			start = i*n + 1
			end = start + n -1
			ub[start:end] = 1 / (n*(1-self.alphas[i]))
			A[i, start:end] = 1

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

		etas = np.zeros(q*n)
		qp.getPrimalSolution(etas)
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
			if len(hyperSVidx)==0:
				hyperSVidx = np.arange(len(eta))[eta > self.tol]
				rhos[j] = max(np.dot(H[hyperSVidx][:,SVidx], etastars[SVidx])/q)
			else:
				rhos[j] = np.median(np.dot(H[hyperSVidx][:,SVidx], etastars[SVidx])/q)
		self.rhos = rhos
		self.etastars = etastars
		tmp = np.dot(H[:,SVidx], etastars[SVidx])/q
		self.rhobounds = tmp.max(), tmp.min()
		return self
		
	def transform(self, X, interpolate=False):
		q = len(self.rhos)
		if self.kernel == 'gauss':
			K = self.kernel_fun(self.X, X, gamma=self.gamma)
		else:
			K = self.kernel_fun(self.X, X)
		objFun = np.dot(K, self.etastars)/q
		out = np.ones(X.shape[0])
		if interpolate:
			for i, x in enumerate(objFun):
				if x >= max(self.rhos+self.tol):
					c = max(self.rhobounds[0] - x, 0)/(self.rhobounds[0] - max(self.rhos))
					out[i] = c * min(self.alphas)
				elif x <= min(self.rhos+self.tol):
					c = max(x - self.rhobounds[1], 0)/(min(self.rhos) - self.rhobounds[1])
					out[i] = (1-c) + c * max(self.alphas)
				else:
					tmp = np.arange(len(self.rhos))[self.rhos+self.tol<x]
					c = (x - self.rhos[tmp[0]])/(self.rhos[tmp[0]-1] - self.rhos[tmp[0]])
					out[i] = (1-c) * self.alphas[tmp[0]] + c * self.alphas[tmp[0]-1]
			return out
		out = [(objFun>rho+self.tol).astype(int) for rho in self.rhos]
		out = np.transpose(np.asarray(out))
		return out

