# Density Estimation Forests in Python
## Using Kernel Density Estimation in Random Forests

### Introduction


_In probability and statistics, density estimation is the construction of an estimate, based on observed data, of an unobservable underlying probability density function. The unobservable density function is thought of as the density according to which a large population is distributed; the data are usually thought of as a random sample from that population._


_Random forests are an ensemble learning method that operate by constructing a multitude of decision trees and combining their results for predictions. Random decision forests correct for decision trees' habit of overfitting to their training set._

In this project, a random forests method for density estimation is implemented in python. Following is the presentation of some of the steps, results, tests, and comparisons.


### Random Forest Implementation

In this implementation, axis aligned split functions are used (called [stumps](https://en.wikipedia.org/wiki/Decision_stump)) to build binary trees by optimizing the [entropy gain](https://en.wikipedia.org/wiki/Differential_entropy) at each node. The key parameters to select for this method are: tree depth/entropy gain threshold, forest size, and randomness. 

The optimal depth of a tree will be case dependent. For that reason we first train a small set of trees on a fixed depth (_tune\_entropy\_threshold_ method, parameters _n_ and _depth_). Unlike forest size, where an increase will never yield worse results, a lax stop condition will lead to [overfitting](https://en.wikipedia.org/wiki/Overfitting). The entropy gain is strictly decreasing with depth, as can be seen in the animation below:
<p align="center">
<img src="result_plots/lcurve.gif" width="500px"/>
</p>

Optimizing the entropy gain threshold is an [ill-posed regularization problem](https://en.wikipedia.org/wiki/Tikhonov_regularization), that is handled in this implementation by finding the elbow point of 'maximum depth' (point furthest from the line connecting the the function's extremes), and averaging it out over _n_, as we can see here: 


<p align="center">
<img src="result_plots/evol.png" width="400px"/>
</p>

This step is expensive, since the depth is fixed with no a priori indication of where the optimal threshold is, and the number of leafs that need to be fitted grows exponentially. A better approach would be to implement an online L-curve method (such as the ones discussed [here](https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf)) as a first pass to avoid initial over-splitting (pending).

From _[Decision Forests for Classification,
Regression, Density Estimation, Manifold
Learning and Semi-Supervised Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/decisionForests_MSR_TR_2011_114.pdf)_: 

_A key aspect of decision forests is the fact that its component trees are all randomly different from one another. This leads to de-correlation between the individual tree predictions and, in turn, to improved generalization. Forest randomness also helps achieve high robustness with respect to noisy data.Randomness is injected into the trees during the training phase. Two of the most popular ways of doing so are:_
* _[Random training data set sampling (e.g. bagging) ](https://en.wikipedia.org/wiki/Bootstrap_aggregating)_
* _[Randomized node optimization](https://en.wikipedia.org/wiki/Random_subspace_method)_
_These two techniques are not mutually exclusive and could be used together._

The method is tested by sampling a combination of gaussians. In order to introduce randomness the node optimization is randomized by parameter _rho_, with the available parameter search space at each node split proportional to it. With a 50% _rho_ and 5 trees, we see the firsts results below:


<p align="center">
<img src="result_plots/density_estimation.png" width="400px"/>
</p>

Performance is harder to measure when using random forests for density estimation (as opposed to regression or classification) since we're in the unsupervised space. Here, the [Jensen-Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) is used as a comparison metric (whenever test data from a known distribution is used).

<p align="center">
<img src="result_plots/density_comp.png" width="400px"/>
</p>


### Leaf prediction using KDE 

One of the main problems of [Kernel Density Estimation](https://web.as.uky.edu/statistics/users/pbreheny/621/F10/notes/10-28.pdf) is the choice of bandwidth. Many of the approaches to find it rely on assumptions of the underlying distribution, and perform poorly on clustered, real-world data (although there are methods that incroporate an [adaptive bandwidth](https://indico.cern.ch/event/548789/contributions/2258640/attachments/1327011/1992522/kde.pdf) effectively).

The module can work with any imlementation of the _Node_ class. In these first examples the _NodeGauss_ class is used, by fitting a gaussian distribution at each leaf. Below can be seen the results of using _NodeKDE_, where the compactness measure is still based on the gaussian differential entropy, but the leaf prediction is the result of the  KDE method. By looking for splits that optimize fitting a gaussian function, many of the multivariate bandwidth problems that KDE has are avoided, and Silverman's rule for bandwidth selection can be used with good results:

<p align="center">
<img src="result_plots/density_comp_kde.png" width="400px"/>
</p>


Although it produces an overall much better JSD, it's worth noting that the top right 'bump' is overfitting the noise more. This is expected since the underlying distribution of our test data is a combination of Gaussians, and if a leaf totally encompasses a bump (as can be seen in the forest representation below), then fitting a Gaussian function will perform better than any non parametric technique. 

<p align="center">
<img src="result_plots/combined_trees.png" width="400px"/>
</p>


#### To do

* Try other entropy gain functions / compactness measures.
* Use online L-curve method for entropy threshold optimization.
* Other bottlenecks.
* Refactor to reuse framework in classification and regression.



