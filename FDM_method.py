import numpy as np
from numpy import linalg as LA
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from typing import Callable, List, Any

class Poisson:
    '''
    Solves Poisson partial differential equation defined on rectangle.
    '''
    def __init__(self, f: Callable[[float, float], float], 
                 lowerU: Callable[[float], float],
                 upperU: Callable[[float], float],
                 leftU: Callable[[float], float],
                 rightU: Callable[[float], float]) -> None:
        '''
        Initiation of Poisson partial differential equation on rectangle. 
        
        Args:
            f (Callable[[float ,float], float]): function f
            lowerU (Callable[[float], float]): lower boundary of rectangle
            upperU (Callable[[float], float]): upper boundary of rectangle
            leftU (Callable[[float], float]): left boundary of rectangle
            rightU (Callable[[float], float]): right boundary of rectangle'''
        self.f: Callable[[float ,float], float] = f
        self.lowerU: Callable[[float], float] = lowerU
        self.upperU: Callable[[float], float] = upperU
        self.leftU: Callable[[float], float] = leftU
        self.rightU: Callable[[float], float] = rightU

        self.solution: Callable[[float ,float], float]|None = None
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

    def setNM(self, n : int, m : int) -> None:
        '''
        Sets the number of points.
        
        Args:
            n (int): number of points on x axis
            m (int): number of points on y axis'''
        self.n = n
        self.m = m
        self.set_n = True
        self.setHK()

    def setLabel(self, label : str) -> None:
        '''
        Sets given equation in LaTeX to write it on plot.
        
        Args:
            label (str): label of graph'''
        self.label = label

    def setInterval(self, x : list, y : list) -> None:
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

    def addSolution(self, solution) -> None:
        '''Add solution to the problem'''
        self.solution = solution
        self.sol_exist = True
    def solve(self) -> np.ndarray:
        '''
        Solves the problem.
        
        Returns:
            NDArray[np.float64]: solution'''
        if not all([self.set_interval, self.set_n]):
            raise ValueError("Cannot solve equation without setting parameters")
        T: NDArray[np.float64] = np.diag(-2*(1/self.h**2+1/self.k**2)*np.ones(self.n)) + np.diag(1/self.h**2*np.ones(self.n-1),-1) + np.diag(1/self.h**2*np.ones(self.n-1),1)
        B: NDArray[np.float64] = np.diag(1/self.k**2*np.ones(self.n))
        A: NDArray[np.float64] = np.kron(np.diag(np.ones(self.m)),T) + np.kron(np.diag(np.ones(self.m-1),-1),B) + np.kron(np.diag(np.ones(self.m-1),1),B)
        F: List[NDArray[np.float64]] = []
        values: float
        for j in range(self.m):
            for i in range(self.n):
                values = self.f(self.unknown_x[i],self.unknown_y[j])
                if j == 0:
                    values -= self.lowerU(self.unknown_x[i])/self.k**2
                elif j == self.m-1:
                    values -= self.upperU(self.unknown_x[i])/self.k**2
                if i == 0:
                    values -= self.leftU(self.unknown_y[j])/self.h**2
                elif i == self.n-1:
                    values -= self.rightU(self.unknown_y[j])/self.h**2
                F.append(values)
        F: NDArray[np.float64] = np.array(F)
        U: NDArray[np.float64] = LA.solve(A,F)
        U = U.reshape([self.m,self.n])
        U = np.flipud(U)
        return U
    def plt_config(self) -> None:
        '''
        Configures plot.
        '''
        A: int = 6
        plt.rc('figure', figsize=[90* .5**(.5 * A), 45.11 * .5**(.5 * A)])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{polski}')
        plt.rc('font', family='serif')
    def show(self) -> float|None:
        '''
        Plots the solution.
        
        Returns:
            str: Error norm if can.'''
        self.plt_config()
        U: NDArray[np.float64] = self.solve()
        X: NDArray[np.float64]
        Y: NDArray[np.float64]
        X, Y = np.meshgrid(self.x,self.y)

        lowBoundX: NDArray[np.float64] = np.array(list(map(self.lowerU,self.x)))
        upBoundX: NDArray[np.float64] = np.array(list(map(self.upperU,self.x)))
        lowBoundY: NDArray[np.float64] = np.array(list(map(self.leftU,self.y)))
        upBoundY: NDArray[np.float64] = np.array(list(map(self.rightU,self.y)))

        U = np.vstack((U,lowBoundX[1:len(self.x)-1]))
        U = np.vstack((upBoundX[1:len(self.x)-1],U))

        U = np.hstack((np.flipud(lowBoundY.reshape([self.m+2,1])),U))
        U = np.hstack((U,np.flipud(upBoundY.reshape([self.m+2,1]))))

        fig: plt.Figure = plt.figure()
        ax: plt.Axes
        if self.sol_exist:
            gs = GridSpec(2,2,figure=fig)
            ax = fig.add_subplot(gs[0, 0],projection='3d')
            ax1: plt.Axes = fig.add_subplot(gs[1,0],projection='3d')
            ax1.plot_wireframe(X,Y,self.solution(X,Y), label="Rozwiązanie dokładne", color="#00853f")
            ax1.set_title("Rozwiązanie dokładne")
            ax2: plt.Axes = fig.add_subplot(gs[:,1],projection='3d')
            ax2.set_title("Błąd")
            surf: plt.Axes = ax2.plot_surface(X,Y,np.abs(self.solution(X,Y)-np.flipud(U)),cmap="Reds",
                       linewidth=0, antialiased=False, label="Błąd")
            fig.colorbar(surf, shrink=0.5, aspect=10)
        else:
            ax = fig.add_subplot(projection='3d')
        ax.plot_wireframe(X,Y,np.flipud(U), label="Rozwiązanie numeryczne", color="#f68c14")
        ax.set_title("Rozwiązanie numeryczne")
        fig.suptitle(self.label)
        plt.grid(True)
        plt.show()
        if self.sol_exist:
            return LA.norm(self.solution(X,Y)-np.flipud(U), np.inf)

    def __str__(self) -> str:
        '''
        String representation of result. Shows result in plot with simple print function.
        '''
        solution: float|None = self.show()
        if self.sol_exist:
            return f"Bład: {solution}"
        return f""
    
def main() -> None:
    #Równanie Różniczkowe cząstkowe rzędu II typu eliptycznego
    f: Callable[[float, float], float] = lambda x, y: 0
    #Warunki brzegowe
    lowerU: Callable[[float], float] = lambda x: 0
    upperU: Callable[[float], float] = lambda x: x
    leftU: Callable[[float], float] = lambda y: 0
    rightU: Callable[[float], float] = lambda y: y
    #Przedziały x i y
    x0: float = 0
    xn: float = 1
    y0: float = 0
    yn: float = 1
    #Rozwiązanie dokładne
    udok: Callable[[float, float], float] = lambda x,y: x*y

    n: int = 10   #Liczba węzłów wewnętrzych dla x
    m: int = 10  #Liczba węzłów wewnętrzych dla y

    poisson: Poisson = Poisson(f, lowerU, upperU, leftU, rightU)
    poisson.setNM(n,m)
    poisson.setInterval([x0,xn], [y0,yn])
    poisson.addSolution(udok)
    print(poisson)

if __name__ == '__main__':
    main()