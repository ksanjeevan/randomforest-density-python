

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from df_help import *

from grid import Grid
from tree import Tree



def gauss_entropy_func(S):
    """
    Gaussian differential entropy (ignoring constant terms since 
    we're interested in the delta).
    """
    return np.log(np.linalg.det(np.cov(S, rowvar=False)))


class DensityForest:
    """
    Class for density forest, compute entropy threshold for stop condition, then build and
    train forest of trees with randomness rho.
    """

    def __init__(self, data, f_size):
        self.data = data
        self.f_size = f_size

        self.entropy_func = gauss_entropy_func

        self.grid_obj = Grid(data, 100)

        self.grid = self.grid_obj.axis
        
        self.rho = .5

        self.opt_entropy = self.tune_entropy_threshold(plot_debug=True)
        self.forest = self.build_forest()




    # Normalization
    # Prediction




    def build_forest(self):

        forest = {}

        for t in range(self.f_size):
            forest[t] = Tree(self)
            forest[t].tree_plot_leafs(fname='tree_opt%s.png'%t)

            print([x.leaf for x in forest[t].tree_nodes_domain[max(forest[t].tree_nodes_domain)]])
            print([x.leaf for x in forest[t].leaf_nodes])




        
        return forest




    # Implement Online L-curve optimization like EWMA to get rid of input depth
    def tune_entropy_threshold(self, n=10, depth=4, plot_debug=False):
        """
        Compute mean optimal entropy based on L-curve elbow method.
        """
        
        e_arr = []
        for i in range(n):
            
            var = Tree(self, depth=depth)
            e_arr += [pair + [i] for pair in var.entropy_gain_evol]

        entropy_evol = pd.DataFrame(e_arr, columns=['depth', 'entropy', 'tree'])
        entropy_evol = entropy_evol.groupby(['tree', 'depth'])[['entropy']].mean().reset_index().pivot(columns='tree', index='depth', values='entropy').fillna(0)
        entropy_elbow_cand = entropy_evol.apply(lambda x: opt_L_curve(np.array(x.index), np.array(x)))
        
        if plot_debug:
            
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            entropy_evol.plot(ax=ax, kind='line', alpha=.6, lw=3.)
            plt.savefig('evol.png', format='png')
            plt.close()

        return entropy_elbow_cand.mean()










if __name__ == "__main__":
    '''
    g = np.random.multivariate_normal
    data = g([0,0], [[8,0],[0,8]], 100)
    data2 = g([15,0], [[2,0],[0,2]], 100)
    data3 = g([0,15], [[2,0],[0,2]], 100)
    data4 = g([20,20], [[10,0],[0,10]], 100)
    data5 = g([30,30], [[8,0],[0,8]], 100)
    data6 = g([40,0], [[5,0],[0,5]], 300)

    data = np.array(list(data)+list(data2)+list(data3)+list(data4)+list(data5)+list(data6))
    np.save('data.npy', data)
    '''
    

    data = np.load('data.npy')

    
    foo = DensityForest(data, f_size=1)




    

