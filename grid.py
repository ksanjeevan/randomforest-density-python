
import numpy as np

class Grid(object):
    """
    Class which defines the boundaries and granularity of the finite grid.
    """


    def __init__(self, data, div):
        self.data = data
        self.perc_padding = .2

        self.partitions, self.axis = self.init_grid(div)
        #self.grid, self.step, self.displ = self.init_grid()
        #self.split_map = self.create_split_map()


    def init_grid(self, div):

        maxi, mini = np.max(self.data, axis=0), np.min(self.data, axis=0)
        step = (maxi - mini)/div

        mini = mini - step * div * self.perc_padding
        maxi = maxi + step * div * self.perc_padding

        grid_range = int(div*(1+2*self.perc_padding))
        axis = [[mini[0] + i * step[0] for i in range(grid_range)], [mini[1] + j * step[1] for j in range(grid_range)]]

        return grid_range, axis




class PDFComp:

    def __init__(self, obj1, obj2, grid_obj):
        
        pass












