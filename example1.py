import numpy as np
from scipy.stats import multivariate_normal as mvn
from qocsvm import QOCSVM
import matplotlib.pyplot as plt

np.random.seed(666)

## create samples from a gaussian mixture model
n = 200
rv1 = mvn(mean=[0,0]).rvs(size=n)
rv2 = mvn(mean=[3,3]).rvs(size=n)
choose = np.random.choice([0,1], size=n)
X = np.zeros((n,2))
X[choose==0] = rv1[choose==0]
X[choose==1] = rv2[choose==1]

## estimate i/q% quantile regions
q = 5
alphas = np.arange(1,q)/q
quants = QOCSVM(alphas, gamma=.25)
quants.fit(X)
result = quants.transform(X)

## samples in each set (always bounded by alpha_i)
print(alphas)
print(result.mean(axis=0))

## plot quantile regions
N = 150
grid = [[i,j] for i in range(N) for j in range(N)]
grid = np.asarray(grid)/N
a1, a2 = X.min(axis=0)
b1, b2 = X.max(axis=0)
grid[:,0] = grid[:,0]*(b1-a1) + a1
grid[:,1] = grid[:,1]*(b2-a2) + a2

regions = quants.transform(grid).sum(axis=1)
regions2 = quants.transform(grid, interpolate=True)

plt.scatter(grid[:,0], grid[:,1], c=regions)
plt.scatter(X[:,0],X[:,1], marker='^', s=8, c='black')
plt.scatter([0,3],[0,3], marker='x', s=32, c='red')
plt.show()

plt.scatter(grid[:,0], grid[:,1], c=1-regions2)
plt.scatter(X[:,0],X[:,1], marker='^', s=8, c='black')
plt.scatter([0,3],[0,3], marker='x', s=32, c='red')
plt.show()
