import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA
from typing import Callable
from Poisson_Equation import Equation

class Adaptive_grid(Equation):
    def solve(self) -> NDArray[np.float64]:
        '''
        Solves the problem.
        
        Returns:
            NDArray[np.float64]: solution.'''
        if not all([self.set_interval, self.set_n]):
            raise ValueError("Cannot solve equation without setting parameters")

        U: NDArray[np.float64] = np.transpose(np.zeros((self.n+2,self.m+2)))
        U[0,:] = np.array(list(map(self.lowerU, self.x)))
        U[-1,:] = np.array(list(map(self.upperU, self.x)))
        U[:,0] = np.array(list(map(self.leftU, self.y)))
        U[:,-1] = np.array(list(map(self.rightU, self.y)))

        U = np.transpose(U)
        U1: NDArray[np.float64] = np.copy(U)
        for _ in range(10):
            for j in range(1,self.m+1):
                for i in range(1,self.n+1):
                    U1[i,j] = ((U[i+1,j]+U[i-1,j])/(self.h**2)+(U[i,j+1]+U[i,j-1])/(self.k**2)-(self.f(self.x[i],self.y[j])))/(2*(1/(self.h**2)+1/(self.k**2)))
            U = np.copy(U1)
        h: float = (self.xn-self.x0)/(2*self.n+2)
        k: float = (self.ym-self.y0)/(2*self.m+2)
        x: NDArray[np.float64] = np.linspace(self.x0, self.xn, 2*self.n+3)
        y: NDArray[np.float64] = np.linspace(self.y0, self.ym, 2*self.m+3)
        x1: NDArray[np.float64]  = x[1:-1]
        y1: NDArray[np.float64]  = y[1:-1]
        nU: NDArray[np.float64]  = np.transpose(np.zeros((2*self.n+3,2*self.m+3)))
        nU[0,:] = np.array(list(map(self.lowerU, x)))
        nU[-1,:] = np.array(list(map(self.upperU, x)))
        nU[:,0] = np.array(list(map(self.leftU, y)))
        nU[:,-1] = np.array(list(map(self.rightU, y)))
        nU = np.transpose(nU)

        for i in np.arange(2,2*self.n+1,2):
            d: int = 1
            t: int = 1
            for j in np.arange(2,2*self.m+1,2):
                nU[i,j] = U[d,t]
                t += 1
            d += 1

        for i in np.arange(1,2*self.n,2):
            for j in np.arange(1,2*self.m,2):
                nU[i,j]=(nU[i-1,j-1]+nU[i-1,j+1]+nU[i+1,j-1]+nU[i+1,j+1])/4
                nU[i+1,j]=(nU[i+1,j+1]+nU[i+1,j-1])/2
                nU[i,j+1]=(nU[i-1,j+1]+nU[i+1,j+1])/2
        for i in range(1,2*self.n+2):
            nU[i,2*self.m+1] = (nU[i-1,2*self.m]+nU[i-1,2*self.m+2]+nU[i+1,2*self.m]+nU[i+1,2*self.m+2])/4
        for i in range(1,2*self.m+2):
            nU[2*self.n+1,i] = (nU[2*self.n,i-1]+nU[2*self.n+2, i-1]+nU[2*self.n, i+1]+nU[2*self.n+2, i+1])/4
        nU1: NDArray[np.float64]  = np.copy(nU)
        lam: float = (1/4)*(np.cos(np.pi/(2*self.n+3))+np.cos(np.pi/(2*self.m+3)))**2
        w: float =1+(lam/(1+np.sqrt(1-lam))**2)
        e: int = 10
        self.temp: int = 0
        while self.temp <= self.numOfIt and e >= self.err:
            for j in range(1,2*self.m+2):
                for i in range(1,2*self.n+2):
                    nU1[i,j] = w*((nU[i+1,j]+nU1[i-1,j])/(h**2)+(nU[i,j+1]+nU1[i,j-1])/(k**2))/(2*(1/(h**2)+1/(k**2)))+(1-w)*nU[i,j]-((self.f(x[i],y[j])))/(2*(1/(h**2)+1/(k**2)))
            e = LA.norm(nU1-nU, np.inf)
            nU = np.copy(nU1)
            self.temp += 1
        self.x = x
        self.y = y
        print(f"Policzono w {self.temp-1} kroków")
        return nU
    
def main() -> None:
    #Równanie Różniczkowe cząstkowe rzędu II typu eliptycznego
    f: Callable[[float, float], float] = lambda x, y: 0
    #Warunki brzegowe
    lowerU: Callable[[float], float]  = lambda x: 2*np.log(x)
    upperU: Callable[[float], float]  = lambda x: np.log(x**2+1)
    leftU: Callable[[float], float]  = lambda y: np.log(y**2+1)
    rightU: Callable[[float], float]  = lambda y: np.log(y**2+4)
    #Przedziały x i y
    x0: float = 1
    xn: float = 2
    y0: float = 0
    yn: float = 1
    #Rozwiązanie dokładne
    udok: Callable[[float, float], float]  = lambda x,y: np.log(x**2+y**2)

    n: int = 2   #Liczba węzłów wewnętrzych dla x
    m: int = 2  #Liczba węzłów wewnętrzych dla y

    grid: Adaptive_grid = Adaptive_grid(f, lowerU, upperU, leftU, rightU)
    grid.setNM(n,m)
    grid.setInterval([x0,xn], [y0,yn])
    grid.addSolution(udok)
    grid.setError(10**-10, 100000)
    print(grid)

if __name__ == '__main__':
    main()
