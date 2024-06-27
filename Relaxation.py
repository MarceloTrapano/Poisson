import numpy as np
from numpy import linalg as LA
from numpy.typing import NDArray
from Poisson_Equation import Equation
from typing import Callable


class Relaxation(Equation):
    '''
    Implements Young relaxation method for Poisson partial diffetential equation
    '''
    def solve(self) -> NDArray[np.float64]:
        '''
        Solves the problem.
        
        Returns:
            NDArray[np.float64]: solution'''
        if not all([self.set_interval, self.set_n]):
            raise ValueError("Cannot solve equation without setting parameters")
        U: NDArray[np.float64] = np.zeros([self.n+2,self.m+2])
        U[:,0] = np.array(list(map(self.lowerU, self.x)))
        U[:,-1] = np.array(list(map(self.upperU, self.x)))
        U[0,:] = np.array(list(map(self.leftU, self.y)))
        U[-1,:] = np.array(list(map(self.rightU, self.y)))
        lamb: float = 1/4*(np.cos(np.pi/self.n)+np.cos(np.pi/self.m))**2
        omega: float = 1 + lamb/(1+np.sqrt(1-lamb))**2
        U1: NDArray[np.float64]
        e: np.float64
        for k in range(self.numOfIt):
            U1 = np.copy(U)
            for i in range(1,self.n+1):
                for j in range(1,self.m+1):
                    U1[i,j] = omega*((U[i+1,j]+U1[i-1,j])/self.h**2 + (U[i,j+1]+U1[i,j-1])/self.k**2)/(2*(1/self.h**2+1/self.k**2)) + (1-omega)*U[i,j] - self.f(self.x[i],self.y[j])/(2*(1/self.h**2+1/self.k**2))
            e = LA.norm(U-U1,np.inf)
            e = LA.norm(U-U1,np.inf)
            U = np.copy(U1)
            if e <= self.err:
                print(f"Policzono w {k} kroków")
                self.temp = k
                print(e)
                return U
        self.temp: int = self.numOfIt
        print(e)
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

    relaxation: Relaxation = Relaxation(f, lowerU, upperU, leftU, rightU)
    relaxation.setNM(n,m)
    relaxation.setInterval([x0,xn], [y0,yn])
    relaxation.addSolution(udok)
    relaxation.setError(10**-15, 100000)
    print(relaxation)

if __name__ == '__main__':
    main()
