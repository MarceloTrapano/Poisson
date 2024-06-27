import numpy as np
from numpy import linalg as LA
from numpy.typing import NDArray
from Poisson_Equation import Equation
from typing import Callable

class GaussSeid(Equation):
    '''
    Implements Gauss Seidler method for Poisson partial diffetential equation
    '''
    def solve(self) -> NDArray[np.float64]:
        '''
        Solves the problem.
        
        Returns:
            NDArray[np.float64]: solution'''
        if not all([self.set_interval, self.set_n]):
            raise ValueError("Cannot solve equation without setting parameters")
        U : np.ndarray = np.zeros([self.n+2,self.m+2])
        U[:,0] = np.array(list(map(self.lowerU, self.x)))
        U[:,-1] = np.array(list(map(self.upperU, self.x)))
        U[0,:] = np.array(list(map(self.leftU, self.y)))
        U[-1,:] = np.array(list(map(self.rightU, self.y)))
        U1: NDArray[np.float64]
        e: np.float64
        for _ in range(self.numOfIt):
            U1 = np.copy(U)
            for i in range(1,self.n+1):
                for j in range(1,self.m+1):
                    U1[i,j] = ((U[i+1,j]+U1[i-1,j])/self.h**2 + (U[i,j+1]+U1[i,j-1])/self.k**2 - self.f(self.x[i],self.y[j]))/(2*(1/self.h**2+1/self.k**2))
            e = LA.norm(U-U1,np.inf)
            U = np.copy(U1)
            if e <= self.err:
                print(_)
                self.temp: int = _
                print(e)
                return U
        print(e)
        self.temp: int = self.numOfIt
        return U

    
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
    m: int = 5  #Liczba węzłów wewnętrzych dla y

    gaussseid: GaussSeid = GaussSeid(f, lowerU, upperU, leftU, rightU)
    gaussseid.setNM(n,m)
    gaussseid.setInterval([x0,xn], [y0,yn])
    gaussseid.addSolution(udok)
    gaussseid.setError(10**-15, 100000)
    print(gaussseid)

if __name__ == '__main__':
    main()
