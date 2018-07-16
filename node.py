
import numpy as np
from df_help import *



class Node(object):
    """
    Class for each of the nodes in a decision tree.
    """

    def __init__(self, data, quad, depth):
        
        self.go_right = None
        self.quad = quad
        self.depth = depth
        self.s_l = len(data)

        self.left = None
        self.right = None

        


    def add_split(self, value, axis):
        return lambda x: x[axis] > value

    '''
    def leaf_output(self, x):
        """
        Evaluate the density estimation of that leaf on x.
        """
        gauss_arg = np.inner(np.transpose((x-self.mu)), np.inner(self.inv_cov, (x-self.mu)))
        return (np.exp(-.5*gauss_arg))/(2*np.pi*self.sqrt_cov)
    '''         
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





class NodeGauss(Node):
    """
    Class for each of the nodes in a decision tree.
    """

    def __init__(self, data, quad, depth, leaf=False):
        
        super(NodeGauss, self).__init__(data=data, quad=quad, depth=depth)

        self.leaf = leaf
        if leaf:
            self.cov = np.cov(data, rowvar=False)
            # Check cov positive semidef
            # np.all(np.linalg.eigvals(np.cov(data, rowvar=False)) > 0)
            if np.all(np.linalg.eigvals(np.cov(data, rowvar=False)) > 0):
                self.sqrt_cov = np.sqrt(np.linalg.det(self.cov))
                self.inv_cov = np.linalg.inv(self.cov)
                self.mu = np.mean(data, axis=0)
            else:
                self.sqrt_cov = 1
                self.inv_cov = np.zeros(self.cov.shape)
                self.mu = np.mean(data, axis=0)


    def leaf_output(self, x):
        """
        Evaluate the density estimation of that leaf on x.
        """
        gauss_arg = np.inner(np.transpose((x-self.mu)), np.inner(self.inv_cov, (x-self.mu)))
        return (np.exp(-.5*gauss_arg))/(2*np.pi*self.sqrt_cov)
            


def h_rot(x, d):
    return math.pow(len(x), -(2.0)/(d+4))*np.var(x, axis=0) + 1e-8


class NodeKDE(Node):
    """
    Class for each of the nodes in a decision tree.
    """

    def __init__(self, data, quad, depth, leaf=False):
        
        super(NodeKDE, self).__init__(data=data, quad=quad, depth=depth)
        self.leaf = leaf

        if leaf:

            self.data = data
            h = h_rot(data, len(data[0]))

            self.H = [[h[0], 0],[0, h[1]]]
            '''
            print('-------------------')
            print(data)
            print(data.shape)
            print(self.H)
            '''

            self.H_inv = np.linalg.inv(self.H)
            self.H_inv_sqrt_det = np.sqrt(np.linalg.det(self.H_inv))



    def k_gauss(self, u):
        argum = np.sum(u*np.transpose(np.inner(self.H_inv, u)), axis=1)
        return np.exp(-.5*argum)
    

    def leaf_output(self, x):

        result = self.H_inv_sqrt_det * np.sum(self.k_gauss(x - self.data)) * 1./(2*np.pi*len(self.data))
        return result










if __name__ == "__main__":

    H = np.array([[2,0],[0,2]])
    x = np.array([[1,0],[0,1],[1,1],[0,-1],[-1,0]])

    a = np.inner(np.transpose(x), np.inner(H, x))

    print(np.inner(H, x).shape)
    #np.inner(np.transpose(u), np.inner(self.H_inv, u))













