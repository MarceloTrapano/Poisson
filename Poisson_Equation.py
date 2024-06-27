import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from typing import Callable, List, Any

class Equation:
    '''
    Implements Poisson partial differential equation
    '''
    def __init__(self, f: Callable[[float, float], float],
                lowerU: Callable[[float], float],
                upperU: Callable[[float], float],
                leftU: Callable[[float], float], 
                rightU: Callable[[float], float]) -> None:
        '''
        Initiation of problem with source function f.
        
        Args:
            f (Callable[[float, float], float]): source function f
            lowerU (Callable[[float], float): lower boundary function
            upperU (Callable[[float], float): upper boundary function
            leftU (Callable[[float], float): left boundary function
            rightU (Callable[[float], float): right boundary function'''
        self.f: Callable[[float ,float], float] = f
        self.lowerU: Callable[[float], float] = lowerU
        self.upperU: Callable[[float], float] = upperU
        self.leftU: Callable[[float], float] = leftU
        self.rightU: Callable[[float], float] = rightU

        self.solution: Callable[[float, float], float]|None = None
        self.label: str = ""
        
        self.sol_exist: bool = False
        self.set_n: bool = False
        self.set_interval: bool = False

        self.n: int = 1
        self.m: int = 1
        self.x0: float = 0
        self.xn: float = 1
        self.y0: float = 0
        self.ym: float = 1
        self.k: float = 0
        self.h: float = 0

        self.x: NDArray[np.float64]
        self.y: NDArray[np.float64]
        self.unknown_x: NDArray[np.float64]
        self.unknown_y: NDArray[np.float64]

        self.err: float = 10**-10
        self.numOfIt: int = 1000

    def setNM(self, n: int, m: int) -> None:
        '''
        Sets the number of points.
        
        Args:
            n (int): number of points on x axis
            m (int): number of points on y axis'''
        self.n = n
        self.m = m
        self.set_n = True
        self.setHK()

    def setLabel(self, label: str) -> None:
        '''
        Sets given equation in LaTeX to write it on plot.
        
        Args:
            label (str): label of graph'''
        self.label = label

    def setInterval(self, x: List[float], y: List[float]) -> None:
        '''
        Sets boundary interval.
        
        Args:
            x (List): Two element list containing x boundary.
            y (List): Two element list containing y boundary.'''
        self.x0 = x[0]
        self.xn = x[1]
        self.y0 = y[0]
        self.ym = y[1]
        self.set_interval = True
        self.setHK()
    
    def setHK(self) -> None:
        '''Sets distance between points'''
        self.h = (self.xn - self.x0)/(self.n+1)
        self.k = (self.ym - self.y0)/(self.m+1)
        self.x = np.linspace(self.x0, self.xn, self.n+2)
        self.y = np.linspace(self.y0, self.ym, self.m+2)
        
        self.unknown_x = self.x[1:self.n+1]
        self.unknown_y = self.y[1:self.m+1]
    def setError(self, error: float, iter: int) -> None:
        '''
        Sets tolerance for error and number of iterations.
        
        Args:
            error (float): tolerance for error
            iter (int): number of iterations'''
        self.err = error
        self.numOfIt = iter
    def addSolution(self, solution: Callable[[float, float], float]) -> None:
        '''Add solution to the problem
        
        Args:
            solution (Callable[[float, float], float]): solution for problem'''
        self.solution = solution
        self.sol_exist = True

    def solve() -> None:
        raise NotImplementedError

    def plt_config(self) -> None:
        '''
        Configures plot.
        '''
        A: int = 6
        plt.rc('figure', figsize=[90* .5**(.5 * A), 45.11 * .5**(.5 * A)])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{polski}')
        plt.rc('font', family='serif')
    def show(self) -> float:
        '''
        Plots the solution.
        
        Returns:
            str: Error norm if can.'''
        self.plt_config()
        X: NDArray[np.float64] 
        Y: NDArray[np.float64] 
        U: NDArray[np.float64]  = self.solve()
        X, Y = np.meshgrid(self.x,self.y)
        fig: plt.Figure = plt.figure()
        
        if self.sol_exist:
            gs: Any = GridSpec(2,2,figure=fig)
            ax: plt.Axes = fig.add_subplot(gs[0, 0],projection='3d')
            ax1: plt.Axes = fig.add_subplot(gs[1,0],projection='3d')
            ax1.plot_wireframe(X,Y,self.solution(X,Y), label="Rozwiązanie dokładne", color="#00853f")
            ax1.set_title("Rozwiązanie dokładne")
            ax2: plt.Axes = fig.add_subplot(gs[:,1],projection='3d')
            ax2.set_title("Błąd")
            surf: Any = ax2.plot_surface(X,Y,np.abs(self.solution(X,Y)-np.transpose(U)),cmap="Paired",
                       linewidth=0, antialiased=False, label="Błąd")
            fig.colorbar(surf, shrink=0.5, aspect=10)
        else:
            ax = fig.add_subplot(projection='3d')
        fig.suptitle(self.label)
        ax.plot_wireframe(X,Y,np.transpose(U), label="Rozwiązanie numeryczne", color="#f68c14")
        ax.set_title("Rozwiązanie numeryczne")
        plt.grid(True)
        plt.show()
        if self.sol_exist:
            return LA.norm(self.solution(X,Y)-np.transpose(U), np.inf)
        
    def __str__(self) -> str:
        '''
        String representation of result. Shows result in plot with simple print function.
        '''
        solution: float = self.show()
        if self.sol_exist:
            return f"Bład: {solution}"
        return f""
    