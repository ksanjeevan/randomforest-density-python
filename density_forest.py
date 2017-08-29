

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, errno




def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def integrate_2d(deltas, func):

    suma = 0
    for i in range(len(func)-1):
        for j in range(len(func[0])-1):
            suma += func[i][j] + func[i+1][j] + func[i][j+1] + func[i+1][j+1]
            
    step = 1.
    for d in deltas:
        step *= d

    return 0.25*suma*step


def cartesian(arrays, out=None):

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out



class Grid(object):


    def __init__(self, data, partitions):
        self.data = data
        self.perc_padding = .2

        #self.grid, self.step, self.displ = self.init_grid()
        maxi, mini = np.max(self.data, axis=0), np.min(self.data, axis=0)
        step = (maxi - mini)/partitions

        mini = mini - step * partitions * self.perc_padding
        maxi = maxi + step * partitions * self.perc_padding
        grid_range = int(partitions*(1+2*self.perc_padding))

        self.partitions = grid_range
        self.axis = [[mini[0] + i * step[0] for i in range(grid_range)], [mini[1] + j * step[1] for j in range(grid_range)]]

        #self.split_map = self.create_split_map()

    def create_split_map(self):

        shift = int(self.perc_padding * self.partitions)

        split_map_arr = [np.array(x).astype(int) for x in (self.data - self.displ) / self.step]
        split_map_arr = [tuple(shift + x) for x in split_map_arr]

        split_map = {}

        for k, s in enumerate(split_map_arr):
            if s in split_map:
                split_map[s].append(k)
            else:
                split_map[s] = [k]

        return split_map


    def init_grid(self):

        maxi, mini = np.max(self.data, axis=0), np.min(self.data, axis=0)
        step = (maxi - mini)/self.partitions
        grid_range = int(self.partitions*(1+2*self.perc_padding))

        displ = mini

        mini = mini - step * self.partitions * self.perc_padding
        maxi = maxi + step * self.partitions * self.perc_padding
    
        grid = []

        for i in range(grid_range):

            grid.append([])

            for j in range(grid_range):
                
                x = mini[0] + i * step[0]
                y = mini[1] + j * step[1] 

                grid[i].append( np.array([x,y]) )

        grid = np.array(grid)                

        return grid, step, displ




class Node:

    def __init__(self, data, quad, depth, leaf=False):
        self.leaf = leaf
        self.data = data
        self.quad = quad
        self.depth = depth

        self.left = None
        self.right = None

        self.cov = np.cov(data, rowvar=False)
        # Check cov positive semidef
        #print(np.all(np.linalg.eigvals(np.cov(self.data, rowvar=False)) > 0))

        self.sqrt_cov = np.sqrt(np.linalg.det(self.cov))
        self.inv_cov = np.linalg.inv(self.cov)
        self.mu = np.mean(data, axis=0)


    def leaf_output(self, x):

        gauss_arg = np.inner(np.transpose((x-self.mu)), np.inner(self.inv_cov, (x-self.mu)))
        return (np.exp(-.5*gauss_arg))/(2*np.pi*self.sqrt_cov)

            

    def check_norm(self, grid_axis):

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

        if not (integral > 0.99 and integral < 1.01):
            print('Node of depth %s, norm = %s'%(self.depth, integral))







class DensityForest:


    def __init__(self, data):
        self.data = data
        self.grid = Grid(data, 100)
        self.rho = 1.

        self.build_forest()
        


    def build_forest(self):
        self.tree = Tree(data, self.grid, rho=self.rho)
        #self.tree.check_results()
        self.tree.check_progress()
        exit()
        for i in range(50):
            var = Tree(data, self.grid, rho=self.rho)
            var.check_results(fname=os.getcwd()+'/plots/data_%s.png'%i)



class Tree:

    def __init__(self, data, grid_obj, rho):
        
        self.grid = grid_obj
        self.rho = rho
        self.data = data
        self.leaf_nodes = []

        
        node = self.build_tree()
        self.tree_nodes = self.extract_levels(node)


    def _compute_det_lamb(self, S):

        if S.shape[0] > 2:
            return np.log(np.linalg.det(np.cov(S, rowvar=False)))

        return float('inf')

    def entropy_gain(self, S, ind, axis):
        """
        Compute entropy gain given data set, split index and axis of application
        """

        S_right = S[S[:,axis]>self.grid.axis[axis][ind]]
        S_left = S[S[:,axis]<self.grid.axis[axis][ind]]

        right_entropy = self._compute_det_lamb(S_right)*len(S_right)/len(S)
        left_entropy = self._compute_det_lamb(S_left)*len(S_left)/len(S)

        return self._compute_det_lamb(S) - (left_entropy + right_entropy)


    def build_tree(self):
        quad = [[0,self.grid.partitions-1]]*2
        root_node = self.split_node(quad=quad, depth=0)
        return root_node

    def split_node(self, quad, depth):
        """
        Recursively split nodes until stop condition is reached
        """
        def get_quad(old_quad, axis, opt_ind):
            """
            quad: Return 2*d - indexes that delimit branch domain.
            Splits branch domain based on optimal index and axis of application.
            """
            opt_quad_left = old_quad.copy()
            opt_quad_right = old_quad.copy()

            opt_quad_left[axis] = [old_quad[axis][0], opt_ind]
            opt_quad_right[axis] = [opt_ind, old_quad[axis][1]]

            return opt_quad_left, opt_quad_right
            

        # Compute 2*d masks to get data inside branch domain
        right = self.data[:,0] > self.grid.axis[0][quad[0][0]]
        left = self.data[:,0] < self.grid.axis[0][quad[0][1]]
        top = self.data[:,1] > self.grid.axis[1][quad[1][0]]
        bottom = self.data[:,1] < self.grid.axis[1][quad[1][1]]

        local_data = self.data[(right)&(left)&(top)&(bottom)]

        # Stop Condition
        if depth == 4:
            leaf_node = Node(data=local_data, quad=quad, depth=depth, leaf=True)
            self.leaf_nodes.append( leaf_node )
            return leaf_node

        # d axis ranges inside branch domain
        x_edge = range(quad[0][0], quad[0][1]+1)
        y_edge = range(quad[1][0], quad[1][1]+1)
        
        # Apply randomness rho factor to limit parameter space search
        edge = np.array([(z, 0) for z in x_edge] + [(z, 1) for z in y_edge])
        size = len(edge)
        ind_array = edge[np.random.choice(size, size=int(size*self.rho), replace=False)]
            

        # Find split with maxiumum entropy gain
        max_entropy = - float('inf')
        opt_ind = -1

        
        
        for ind, axis in ind_array:
            entropy = self.entropy_gain(local_data, ind, axis)

            if entropy > max_entropy:
                max_entropy = entropy
                opt_ind, opt_axis = (ind, axis)
   

        # Split node's quad
        node = Node(data=local_data, quad=quad, depth=depth)
        opt_quad_left, opt_quad_right = get_quad(quad, opt_axis, opt_ind)
        
        node.left = self.split_node(quad=opt_quad_left, depth=depth+1)
        node.right = self.split_node(quad=opt_quad_right, depth=depth+1)

        return node


    def extract_levels(self, node):

        if node.left:

            levels_dic_left = self.extract_levels(node.left)
            levels_dic_right = self.extract_levels(node.right)

            for k, v in levels_dic_right.items():
                if k in levels_dic_left:
                    levels_dic_left[k] += v
                else:
                    levels_dic_left[k] = v

            levels_dic_left[node.depth] = [node]

            return levels_dic_left

        else:
            return {node.depth : [node]}




    def check_progress(self):
        path = os.getcwd() + '/evol/'
        mkdir_p(path)


        
        for d in np.arange(len(self.tree_nodes)):

            nodes = self.tree_nodes[d]
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            for n in nodes:

                n.check_norm(self.grid.axis)
                [[i1, i2], [j1, j2]] = n.quad
                x1, x2 = self.grid.axis[0][i1], self.grid.axis[0][i2]
                y1, y2 = self.grid.axis[1][j1], self.grid.axis[1][j2]                
                ax.fill_between([x1,x2], y1, y2, alpha=.7)

            pd.DataFrame(data, columns=['x', 'y']).plot(ax=ax, x='x', y='y', kind='scatter', lw=0, alpha=.6, s=20, c='k')
            plt.savefig(path + 'branches_depth%s.png'%d, format='png')
            plt.close()






    def check_results(self, fname='data.png'):
        node = self.build_tree()

        self.tree_nodes = self.extract_levels(node)

        print(self.tree_nodes)


        exit()


        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        
        for n in self.leaf_nodes:


            n.check_norm(self.grid.axis)

            [[i1, i2], [j1, j2]] = n.quad
            x1, x2 = self.grid.axis[0][i1], self.grid.axis[0][i2]
            y1, y2 = self.grid.axis[1][j1], self.grid.axis[1][j2]
            
            ax.fill_between([x1,x2], y1, y2, alpha=.7)
 
    
        pd.DataFrame(data, columns=['x', 'y']).plot(ax=ax, x='x', y='y', kind='scatter', lw=0, alpha=.6, s=20, c='k')
        plt.savefig(fname, format='png')
        plt.close()
        






if __name__ == "__main__":
    '''
    data = np.random.multivariate_normal([0,0], [[8,0],[0,8]], 100)
    data2 = np.random.multivariate_normal([15,0], [[2,0],[0,2]], 100)
    data3 = np.random.multivariate_normal([0,15], [[2,0],[0,2]], 100)
    data4 = np.random.multivariate_normal([20,20], [[10,0],[0,10]], 100)
    data5 = np.random.multivariate_normal([30,30], [[8,0],[0,8]], 100)
    data6 = np.random.multivariate_normal([40,0], [[5,0],[0,5]], 300)

    data = np.array(list(data)+list(data2)+list(data3)+list(data4)+list(data5)+list(data6))
    np.save('data.npy', data)
    '''
    

    data = np.load('data.npy')


    




    
    foo = DensityForest(data)




    

