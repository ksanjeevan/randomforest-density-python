# Random Forests for Density Estimation
## Subtitle


### Introduction

From [Decision Forests for Classification,
Regression, Density Estimation, Manifold
Learning and Semi-Supervised Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/decisionForests_MSR_TR_2011_114.pdf):

####_To do_

1) Make sure _output_ function is working correctly.

2) Find numeric approximation for the cumulative multivariate normal distribution function in order to compute Z<sub>t</sub> (avoid numeric $\Delta$ integration if possible).

3) Forest voting and global normalization check.

4) Test case comparison: build JSD computation.

5) Write an exhaustive documentation!

####_Extras_

* Try other entropy gain functions / compactness measures.
* Use online L-curve method for entropy threshold optimization.
* Other bottlenecks.
* Refactor to reuse framework in classification and regression.
  



### Result Example


<p align="center">
<img src="evol.png" width="400px"/>
</p>



