
import os, errno
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

from grid import Grid

def mkdir_p(path):
    """
    Create directory given path.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def integrate_2d(deltas, func):
    """
    2D Integral numeric approximation.
    """

    suma = 0
    for i in range(len(func)-1):
        for j in range(len(func[0])-1):
            suma += func[i][j] + func[i+1][j] + func[i][j+1] + func[i+1][j+1]
            
    step = 1.
    for d in deltas:
        step *= d

    return 0.25*suma*step


def cartesian(arrays, out=None):
    """
    Compute cartesian product between vector set.
    """

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




def opt_L_curve(xs, ys):
    """
    Generic approach to find optimal Y for L-curve.
    """

    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]
    ra = float(y0 - y1) / (x0 - x1)
    rb = y1 - ra*x1
    result = []
    for xp, yp in zip(xs, ys):
        da = -1./ra
        db = yp - da * xp
        x_star = float(db-rb)/(ra-da)
        y_star = ra*x_star + rb
        result.append( [np.sqrt((xp-x_star)**2 + (yp-y_star)**2), xp, yp] )

    return max(result, key=lambda x: x[0])[2]









import abc

class TestData(abc.ABC):

    """
    Abstract Data and Distribution test
    """

    def __init__(self):
        self.dist = []

    @abc.abstractmethod
    def generate_data(self, fname=''):
        pass

    @abc.abstractmethod
    def check_norm(self):
        pass

    @abc.abstractmethod
    def compute_distribution(self):
        pass


    @abc.abstractmethod
    def evaluate(self, x):
        pass






class TestDataGauss(TestData):

    def __init__(self, params, fname, replace=False, partitions=100):

        self.replace = replace
        self.mu = params['mu']
        self.cov = params['cov']
        self.n = params['n']
        self.Z = [N/np.sum(self.n) for N in self.n]

        self.data = self.generate_data(fname=fname)
        self.grid_obj = Grid(self.data, partitions)
        self.grid = self.grid_obj.axis
        

        self.dist = self.compute_distribution()

     
    def check_norm(self):
        dist_vals = []

        deltas = []

        for v in self.grid:
            deltas.append( v[1]-v[0] )

        for i, x in enumerate(self.grid[0]):
            dist_vals.append([])
            for j, y in enumerate(self.grid[1]):
                dist_vals[i].append(self.evaluate(np.array([x, y])))

        integral = integrate_2d(deltas=deltas, func=dist_vals)

        return integral


    def compute_distribution(self):
        dist = []
        for j, y in enumerate(self.grid[1]):
            dist.append([])
            for i, x in enumerate(self.grid[0]):    
                dist[j].append(self.evaluate(np.array([x, y])))
        return dist

    def evaluate(self, x):
        suma = 0
        for i, args in enumerate(zip(self.mu, self.cov)):
            mu, cov = args
            gauss_arg = np.inner(np.transpose((x-mu)), np.inner(np.linalg.inv(cov), (x-mu)))
            suma += self.Z[i]*(np.exp(-.5*gauss_arg))/(2*np.pi*np.sqrt(np.linalg.det(cov)))
        return suma


    def generate_data(self, fname='data.npy'):

        if os.path.isfile(fname) and not self.replace:
            return np.load(fname)
        else:
            g = np.random.multivariate_normal
            data = []

            for mu, cov, n in zip(self.mu, self.cov, self.n):
                data += list(g(mu, cov, n))

            data = np.array(data)

            np.save(fname, data)

            return data


    def check_plot(self):

        X = self.grid[0]
        Y = self.grid[1]
        Z = self.dist
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
   
        vmin = np.min(Z)
        vmax = np.max(Z)
        var = plt.pcolormesh(np.array(X),np.array(Y),np.array(Z), cmap=cm.Greens, vmin=vmin, vmax=vmax)
        plt.colorbar(var, ticks=np.arange(vmin, vmax, (vmax-vmin)/8))
        ax = plt.gca()
        gris = 200.0
        ax.set_facecolor((gris/255, gris/255, gris/255))
        
        ax.scatter(*zip(*self.data), alpha=.5, c='k', s=10., lw=0)

        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        plt.grid()
        fig.savefig('true_dist_check.png', format='png')
        plt.close()




class TestDataAny(TestData):

    def __init__(self, fname, partitions=100):

        self.data = self.generate_data(fname=fname)

        self.grid_obj = Grid(self.data, partitions)
        self.grid = self.grid_obj.axis
        
        self.dist = self.compute_distribution()

        self.check_plot()

    def generate_data(self, fname='data.npy'):

        if os.path.isfile(fname):
            return np.load(fname)
        else:
            raise ValueError('Enter valid path to source data.')

    def check_plot(self):

        X = self.grid[0]
        Y = self.grid[1]
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
   
        ax.scatter(*zip(*self.data), alpha=.5, c='k', s=10., lw=0)

        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        plt.grid()
        fig.savefig('init_data.png', format='png')
        plt.close()


    def check_norm(self):
        pass

    def compute_distribution(self):
        pass

    def evaluate(self, x):
        pass








class CompareDistributions:

    def __init__(self, original, estimate):
        self.P = np.array(original.dist)
        self.Q = np.array(estimate.dist)
        self.grid = original.grid
        self.data = original.data


    def compute_JSD(self):

        if self.P.shape == self.Q.shape:
            suma = 0

            import math
        
            for i in range(len(self.P)):
                for j in range(len(self.P[0])):    
                    if self.P[i][j] > 0:
                        suma += self.P[i][j]*np.log(2.*self.P[i][j]/(self.P[i][j] + self.Q[i][j]))
                    
            for i in range(len(self.Q)):
                for j in range(len(self.Q[0])):
                    if self.Q[i][j] > 0:
                        suma += self.Q[i][j]*np.log(2.*self.Q[i][j]/(self.P[i][j] + self.Q[i][j]))

            return suma



    def vizualize_both(self, fname='density_comp.png', show_data=False):
        X = self.grid[0]
        Y = self.grid[1]
        Z1 = self.P
        Z2 = self.Q

        fig = plt.figure(figsize=(12, 12))

        true_dist_params = (211, Z1, 'True Distribution', cm.Greens)
        rf_dist_params = (212, Z2, 'Density Forest Estimate; JSD = %.3f'%(self.compute_JSD()), cm.Blues)

        for sp, Z, title, cmap in [true_dist_params, rf_dist_params]:
            ax = fig.add_subplot(sp)
            vmin=np.min(Z1)
            vmax=np.max(Z1)
            var = plt.pcolormesh(np.array(X),np.array(Y),np.array(Z), cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(var, ticks=np.arange(vmin, vmax, (vmax-vmin)/8))
            ax = plt.gca()
            gris = 200.0
            ax.set_facecolor((gris/255, gris/255, gris/255))
            
            if show_data:
                ax.scatter(*zip(*self.data), alpha=.5, c='k', s=10., lw=0)
            
            ax.set_title(title)
            plt.xlim(np.min(X), np.max(X))
            plt.ylim(np.min(Y), np.max(Y))
            plt.grid()


        fig.savefig(fname, format='png')
        plt.close()




