# q-OCSVM
Implementation of https://papers.nips.cc/paper/5125-q-ocsvm-a-q-quantile-estimator-for-high-dimensional-distributions.

q-OCSVM constructs q hierarchically nested quantile regions at once, with an objective function similar to One-Class-SVMs. It achieves nested sets by enforcing the hyperplanes in kernel space to be parallel. 

Given a set of n samples and distinct values a_i (i=1..q) between 0 and 1, q-OCSVM returns decision functions 1_Ci for sets Ci such that:
  * Ci is a subset of Cj if and only if a_i < a_j
  * |{x in Ci}|/n is at most a_i.

The objective function is a quadratic program of dimension *nq* with a dense hessian, so it is only feasible for small datasets. 

**Requires**: numpy, scipy, [qpOASES](https://projects.coin-or.org/qpOASES)
