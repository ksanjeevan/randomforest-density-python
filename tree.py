

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from df_help import *
from node import Node




class Tree:
    """
    Class for training and building decision trees.
    """

    def __init__(self, forest_obj, depth=None):

        self.forest_obj = forest_obj

        self.leaf_nodes = []
        self.entropy_gain_evol = []
        
        self.explore_depth = depth if depth else 0


        node = self.build_tree()

        self.tree_nodes_depth = self.extract_levels(node)
        self.tree_nodes_domain = self.extract_domain_splits(node)


    def output(self, x):
        current_node = self.root_node
        
        while not current_node.leaf:
            if current_node.go_right(x):
                current_node = current_node.right
            else:
                current_node = current_node.left

        return current_node.leaf_output(x)



    def _compute_det_lamb(self, S):

        if S.shape[0] > 2:
            return self.forest_obj.entropy_func(S)
        return 1e5


    def entropy_gain(self, S, ind, axis):
        """
        Compute entropy gain given data set, split index and axis of application
        """

        S_right = S[S[:,axis]>=self.forest_obj.grid[axis][ind]]
        S_left = S[S[:,axis]<self.forest_obj.grid[axis][ind]]

        #print(S, ind, axis)
        #print(S_right, S_left)

        right_entropy = self._compute_det_lamb(S_right)*len(S_right)/len(S)
        left_entropy = self._compute_det_lamb(S_left)*len(S_left)/len(S)


        a = self._compute_det_lamb(S)
        b = right_entropy
        c = left_entropy
        
        return a - (b + c), len(S_left), len(S_right)


    def build_tree(self):
        quad = [[0,len(self.forest_obj.grid[0])-1]]*2
        root_node = self.split_node(quad=quad, depth=0)
        return root_node



    def _get_local_data(self, quad):
        right = self.forest_obj.data[:,0] >= self.forest_obj.grid[0][quad[0][0]]
        left = self.forest_obj.data[:,0] < self.forest_obj.grid[0][quad[0][1]]
        top = self.forest_obj.data[:,1] >= self.forest_obj.grid[1][quad[1][0]]
        bottom = self.forest_obj.data[:,1] < self.forest_obj.grid[1][quad[1][1]]

        return self.forest_obj.data[(right)&(left)&(top)&(bottom)]

    def _get_search_space(self, quad):
        # d axis ranges inside branch domain
        x_edge = range(quad[0][0], quad[0][1]+1)
        y_edge = range(quad[1][0], quad[1][1]+1)
        
        # Apply randomness rho factor to limit parameter space search
        edge = np.array([(z, 0) for z in x_edge] + [(z, 1) for z in y_edge])
        size = len(edge)
        return edge[np.random.choice(size, size=int(size*self.forest_obj.rho), replace=False)]


    def _find_opt_cut(self, ind_array, local_data):
        max_entropy = 0
        opt_ind = -1
        opt_axis = -1

        for ind, axis in ind_array:

            entropy, left_size, right_size = self.entropy_gain(local_data, ind, axis)

            if entropy > max_entropy and left_size > 2 and right_size > 2:
                max_entropy = entropy
                opt_ind, opt_axis = (ind, axis)

        return max_entropy, opt_ind, opt_axis

    def _get_new_quad(self, old_quad, axis, opt_ind):
        """
        quad: Return 2*d - indexes that delimit branch domain.
        Splits branch domain based on optimal index and axis of application.
        """
        opt_quad_left = old_quad.copy()
        opt_quad_right = old_quad.copy()

        opt_quad_left[axis] = [old_quad[axis][0], opt_ind]
        opt_quad_right[axis] = [opt_ind, old_quad[axis][1]]

        return opt_quad_left, opt_quad_right


    def split_node(self, quad, depth):
        """
        Recursively split nodes until stop condition is reached
        """

        # Restrict data to in branch domain
        local_data = self._get_local_data(quad)

        # Restrict search space for optimal cut
        ind_array = self._get_search_space(quad)
            
        # Find split with maxiumum entropy gain
        
        max_entropy, opt_ind, opt_axis = self._find_opt_cut(ind_array, local_data)

        tune_threshold_cond = depth == self.explore_depth
        stop_condition = tune_threshold_cond if self.explore_depth else (self.forest_obj.opt_entropy > max_entropy)
        
        # Stop Condition
        if stop_condition or opt_ind == -1:
            leaf_node = Node(quad=quad, depth=depth, leaf=True)
            self.leaf_nodes.append( leaf_node )
            return leaf_node

        
        self.entropy_gain_evol.append( [depth, max_entropy] )

        # Split node's quad
        node = Node(quad=quad, depth=depth)
        node.go_right = node.add_split(self.forest_obj.grid[opt_axis][opt_ind], opt_axis)


        opt_quad_left, opt_quad_right = self._get_new_quad(quad, opt_axis, opt_ind)
        
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


    def extract_domain_splits(self, node):

        dic = {}
        count = 0
        dic[count] = [node]
        
        while not all([n.left is None for n in dic[count]]): 

            nodes = dic[count]
            count += 1
            dic[count] = []

            for k, n in enumerate(nodes):
                if n.left:
                    
                    dic[count].append( n.left )
                    dic[count].append( n.right )
                else:
                    dic[count].append( n )
        return dic



    def domain_splits_plots(self):
        path = os.getcwd() + '/evol/'
        mkdir_p(path)
        
        for d in np.arange(len(self.tree_nodes_domain)):

            nodes = self.tree_nodes_domain[d]
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            for n in nodes:

                #n.check_norm(self.grid.axis)
                [[i1, i2], [j1, j2]] = n.quad
                x1, x2 = self.forest_obj.grid[0][i1], self.forest_obj.grid[0][i2]
                y1, y2 = self.forest_obj.grid[1][j1], self.forest_obj.grid[1][j2]                
                ax.fill_between([x1,x2], y1, y2, alpha=.7)

            pd.DataFrame(self.forest_obj.data, columns=['x', 'y']).plot(ax=ax, x='x', y='y', kind='scatter', lw=0, alpha=.6, s=20, c='k')
            plt.savefig(path + 'branches_depth%s.png'%d, format='png')
            plt.close()



    def tree_plot_leafs(self, fname='data.png'):

        path = os.getcwd() + '/plots/'
        mkdir_p(path)


        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        
        for n in self.leaf_nodes:

            #n.check_norm(self.grid.axis)

            [[i1, i2], [j1, j2]] = n.quad
            x1, x2 = self.forest_obj.grid[0][i1], self.forest_obj.grid[0][i2]
            y1, y2 = self.forest_obj.grid[1][j1], self.forest_obj.grid[1][j2]
            
            ax.fill_between([x1,x2], y1, y2, alpha=.7)
 
        pd.DataFrame(self.forest_obj.data, columns=['x', 'y']).plot(ax=ax, x='x', y='y', kind='scatter', lw=0, alpha=.6, s=20, c='k')
        plt.savefig(path + fname, format='png')
        plt.close()
        
