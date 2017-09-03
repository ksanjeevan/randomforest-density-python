
import numpy as np
from df_help import *

class Node:
    """
    Class for each of the nodes in a decision tree.
    """

    def __init__(self, data, quad, depth, leaf=False):
        
        self.go_right = None
        self.leaf = leaf
        self.quad = quad
        self.depth = depth
        self.s_l = len(data)

        self.left = None
        self.right = None


        if leaf:
            self.cov = np.cov(data, rowvar=False)
            # Check cov positive semidef
            #print(np.all(np.linalg.eigvals(np.cov(self.data, rowvar=False)) > 0))
            self.sqrt_cov = np.sqrt(np.linalg.det(self.cov))
            self.inv_cov = np.linalg.inv(self.cov)
            self.mu = np.mean(data, axis=0)

        


    def add_split(self, value, axis):
        return lambda x: x[axis] > value


    def leaf_output(self, x):
        """
        Evaluate the density estimation of that leaf on x.
        """
        gauss_arg = np.inner(np.transpose((x-self.mu)), np.inner(self.inv_cov, (x-self.mu)))
        return (np.exp(-.5*gauss_arg))/(2*np.pi*self.sqrt_cov)
            

    def check_norm(self, grid_axis):
        """
        Verify leaf integrates to ~ 1.
        """

        dist_vals = []

        grid_axis_local = grid_axis.copy()
        grid_axis_local[0] = grid_axis_local[0][self.quad[0][0]:self.quad[0][1]+1]
        grid_axis_local[1] = grid_axis_local[1][self.quad[1][0]:self.quad[1][1]+1]

        deltas = []

        for v in grid_axis_local:
            deltas.append( v[1]-v[0] )

        for i, x in enumerate(grid_axis_local[0]):
            dist_vals.append([])
            for j, y in enumerate(grid_axis_local[1]):
                dist_vals[i].append(self.leaf_output(np.array([x, y])))
        

        integral = integrate_2d(deltas=deltas, func=dist_vals)

        if not (integral > 0.95 and integral < 1.05):
            print('Node of depth %s, norm = %s'%(self.depth, integral))

        return integral
