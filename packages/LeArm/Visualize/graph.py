import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D



class graph:
    # The main class responsible for managing figures. All graphs are placed into a grid setup, to support more complex figure generation.
    def __init__(self, pyplotFigurekwargs=None, figRows=1, figCols=1):
        '''Create a matplotlib.pyplot.figure(), which for practical purposes is the window that appears when using matplotlib.
        Input:
            pyplotFigurekwargs: optional, default:None
                Dict of key word arguments to be handled by matplotlib.pyplot.figure(), see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html for details.
                Note that layout='constrained' is enforced no matter what.
            
            figRows: optional, default: 1
                The number of grid rows to place on the figure, as defined by GridSpec.
            
            figCols: optional, default: 1
                The number of grid columns to place on the figure, as defined by GridSpec.
        '''

        if pyplotFigurekwargs == None:
            self.figure = plt.figure(layout='constrained')
        else:
            if ('constrained' != pyplotFigurekwargs.get('layout')) or (not pyplotFigurekwargs.get('constrained_layout')):
                pyplotFigurekwargs['layout'] = 'constrained'
            self.figure = plt.figure(**pyplotFigurekwargs)

        self.grid = GridSpec(figRows, figCols, figure=self.figure)
        self.figRows = figRows
        self.figCols = figCols

    def create_axes(self, projection='3d', rowNum = 0, colNum = 0, xLim =(-3, 3), yLim=(-3, 3), zLim=(-3, 3), autoScale=False):
        '''Create an axes object, which is an actual graph in a figure.
        Input:
            projection: optional, default: 3d
                Used to specify whether the graph created should be in 3D or 2D.
            
            rowNum: optional, default: 0
                Integer that specifies the row number in the figure where the graph will be placed.

            colNum: optional, default: 0
                Integer that specifies the column number in the figure where the graph will be placed.
            
            xLim: optional, default: (-3, 3)
                Tuple specifying the limits for the x-axis.

            yLim: optional, default: (-3, 3)
                Tuple specifying the limits for the y-axis.
            
            autoScale: optional, default: False
                Boolean flag indicating whether to automatically scale the axes based on the plotted data.
        '''

        if '2d' == projection.lower():
            projection = None # '2d' is not a valid value to call, but None is, and produces a 2D axis
            ax = self.figure.add_subplot(self.grid[rowNum, colNum], projection=projection, aspect='equal', autoscale=autoScale)
            ax.clear()



        else:
            ax = self.figure.add_subplot(self.grid[rowNum, colNum], projection=projection, aspect='equal')
            ax.clear()
            ax.set_aspect('equal')
            ax.set_zlabel('z')
            ax.set_zlim(min(zLim), max(zLim))

        ax.set_xlim(min(xLim), max(xLim))
        ax.set_ylim(min(yLim), max(yLim))
        ax.set_xlabel("x")
        ax.set_ylabel("y")        

        return ax
    
    def parametricCurve3D(self, ax, function, time=(0, 1), resolution=100, xLim=None, yLim=None, update=True, offset=[0, 0, 0]):
        '''
        '''

        xyz = []
        for time in np.linspace(min(time), max(time), resolution):
            xyz.append(np.array(function(time)) + offset)
        
        xyz = np.array(xyz)
        xyz = xyz.transpose()
        
        ax.plot(xyz[0], xyz[1], xyz[2], color='b')

        if update:
            plt.draw()
            plt.pause(0.001)



    
    def mesh(ax, domain, resolution, function, **args):
        ### creates a 3D meshgrid plt
        ### Inputs:
        #   domain:     [xmin, xmax]
        #   resolution: Number of points to evaluate on each axis
        #   function:   lambda function mapping x and y values to z values
        
        # Set limits so that surface appears properly
        ax.set_xlim(domain[0], domain[1])
        ax.set_ylim(domain[0], domain[1])

        increment = (abs(domain[0]) + abs(domain[1]))/resolution
        x = np.arange(domain[0], domain[1], increment)
        y = x

        # Initialize z
        z = [[0] * len(x) for i in range(len(y))]
        # Evaluate function and store meshgrid data
        for i in range(len(x)):
           for i2 in range(len(y)):
               z[i][i2] = function([x[i],y[i2]])

        z = np.array(z)
        ax.set_zlim(np.min(z), np.max(z))

        x, y = np.meshgrid(x, y)

        surf = ax.plot_surface(x, y, z)
        return surf