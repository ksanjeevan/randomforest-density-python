

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from df_help import *

from grid import Grid
from tree import Tree

from pylab import *



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

    def __init__(self, data, grid_obj, f_size, rho=1.):
        self.data = data
        self.f_size = f_size

        self.entropy_func = gauss_entropy_func

        self.grid_obj = grid_obj

        self.grid = self.grid_obj.axis
        
        self.rho = rho

        self.opt_entropy = self.tune_entropy_threshold(plot_debug=True)

        self.forest = self.build_forest()
        self.dist = self.compute_density()

        self.plot_density()



    def compute_density(self):

        dist = []
        for j, y in enumerate(self.grid[1]):
            dist.append([])
            for i, x in enumerate(self.grid[0]):    
                dist[j].append(self.forest_output(np.array([x, y])))
        return dist
        

    
    def plot_density(self):

        X = self.grid[0]
        Y = self.grid[1]
        Z = self.dist

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)


        vmin=np.min(Z)
        vmax=np.max(Z)
        var = plt.pcolormesh(np.array(X),np.array(Y),np.array(Z), cmap=cm.Blues, vmin=vmin, vmax=vmax)
        plt.colorbar(var, ticks=np.arange(vmin, vmax, (vmax-vmin)/8))

        ax = plt.gca()

        gris = 200.0
        ax.set_facecolor((gris/255, gris/255, gris/255))

        ax.scatter(*zip(*self.data), alpha=.5, c='k', s=10., lw=0)

        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        plt.grid()

        ax.set_title('rho = %s, |T| = %d, max_entropy = %.2f'%(self.rho, self.f_size, self.opt_entropy))

        fig.savefig('density_estimation.png', format='png')
        plt.close()






    def forest_output(self, x):

        result = []
        for i, t in self.forest.items():
            result.append( t.output(x) )

        return np.mean(result)



    def build_forest(self):

        forest = {}

        for t in range(self.f_size):
            forest[t] = Tree(self, rho=self.rho)
            forest[t].tree_leaf_plots(fname='tree_opt%s.png'%t)
        
        return forest


    # Implement Online L-curve optimization like EWMA to get rid of input depth
    def tune_entropy_threshold(self, n=5, depth=6, plot_debug=False):
        """
        Compute mean optimal entropy based on L-curve elbow method.
        """
        
        e_arr = []
        for i in range(n):
            
            var = Tree(self, rho=.5, depth=depth)
            e_arr += [pair + [i] for pair in var.entropy_gain_evol]

            var.domain_splits_plots(subpath='%s/'%i)

        entropy_evol = pd.DataFrame(e_arr, columns=['depth', 'entropy', 'tree'])
        entropy_evol = entropy_evol.groupby(['tree', 'depth'])[['entropy']].mean().reset_index().pivot(columns='tree', index='depth', values='entropy').fillna(0)
        entropy_elbow_cand = entropy_evol.apply(lambda x: opt_L_curve(np.array(x.index), np.array(x)))
        
        avg_opt_entropy = entropy_elbow_cand.mean()
        if plot_debug:
            
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            entropy_evol.plot(ax=ax, kind='line', alpha=.6, lw=3., title='Avg. Opt. Entropy = %.2f'%avg_opt_entropy)
            plt.savefig('evol.png', format='png')
            plt.close()

        return avg_opt_entropy











if __name__ == "__main__":
    '''
    
    data = g(, 100)
    data2 = g(, 100)
    data3 = g(, 100)
    data4 = g(, 100)
    data5 = g(, 100)
    data6 = g(, 300)

    data = np.array(list(data)+list(data2)+list(data3)+list(data4)+list(data5)+list(data6))
    np.save('data.npy', data)
    '''

    var = TestDataGauss()
    foo = DensityForest(var.data, grid_obj=var.grid_obj, f_size=5, rho=.5)

    
    tri = CompareDistributions(original=var, estimate=foo)
    
    

    #print(foo.forest[0].output([0, 40]))




    

