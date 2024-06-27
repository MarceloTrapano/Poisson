import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA
from typing import Callable, List, Any
from Poisson_Equation import Equation

class Peaceman(Equation):
    def solve(self) -> NDArray[np.float64]:
        '''
        Solves the problem.
        
        Returns:
            NDArray[np.float64]: solution.'''
        if not all([self.set_interval, self.set_n]):
            raise ValueError("Cannot solve equation without setting parameters")
        temp: int = 0; e : float = 10
        alpha: float = (np.sin(np.pi/(2*np.max([self.m,self.n]))))**2
        pk: int = 1
        while (np.sqrt(2)-1)**(2*pk) >= alpha:
            pk += 1
        pr: list[float] = []
        for i in range(pk):
            pr.append(alpha**((1-(i+1))/(2*pk)))
        r: np.ndarray = np.kron(np.ones(self.numOfIt*self.m*self.n),np.array(pr))

        U: NDArray[np.float64] = np.transpose(np.zeros((self.n+2,self.m+2)))
        U[0,:] = np.array(list(map(self.lowerU, self.x)))
        U[-1,:] = np.array(list(map(self.upperU, self.x)))
        U[:,0] = np.array(list(map(self.leftU, self.y)))
        U[:,-1] = np.array(list(map(self.rightU, self.y)))

        U = np.transpose(U)
        U1: List[float]
        U2: List[float]
        v1: NDArray[np.float64]
        v2: NDArray[np.float64]
        F: List[float]
        N1: NDArray[np.float64]
        N2: NDArray[np.float64]
        A: NDArray[np.float64]
        while temp <= self.numOfIt and e >= self.err:
            U1 = []
            U2 = []
            v1 = (1+2*r[temp])*np.ones(self.m)
            v2 = -r[temp]*np.ones(self.m-1)
            A = np.diag(v1) + np.diag(v2,-1) + np.diag(v2,1)

            for j in range(1,self.n+1):
                F = []
                for i in range(1,self.m+1):
                    F.append(U[j,i] + r[temp]*(U[j+1,i]+U[j-1,i]-2*U[j,i]))
                F[0] += r[temp]*U[j,0]
                F[-1] += r[temp]*U[j,-1]
                F: NDArray[np.float64] = np.array(F)
                N1 = LA.solve(A,F)
                U1.append(N1)
            U1 = np.vstack((U1, U[-1, 1:-1]))
            U1 = np.vstack((U[0,1:-1].T, U1))
            U1 = np.hstack((U1, np.atleast_2d(U[:,-1]).T))
            U1 = np.hstack((np.atleast_2d(U[:,0]).T, U1))

            v1 = (1+2*r[temp])*np.ones(self.n)
            v2 = -r[temp]*np.ones(self.n-1)
            A = np.diag(v1) + np.diag(v2,-1) + np.diag(v2,1)

            for j in range(1,self.m+1):
                F = []
                for i in range(1,self.n+1):
                    F.append(U1[i,j] + r[temp]*(U1[i,j+1]+U1[i,j-1]-2*U1[i,j]))
                F[0] += r[temp]*U1[0,j]
                F[-1] += r[temp]*U1[-1,j]
                F = np.array(F)
                N2 = LA.solve(A,F)
                U2.append(N2)
            U2 = np.transpose(U2)
            U2 = np.vstack((U2, U[-1, 1:-1]))
            U2 = np.vstack((U[0,1:-1], U2))
            U2 = np.hstack((U2, np.atleast_2d(U[:,-1]).T))
            U2 = np.hstack((np.atleast_2d(U[:,0]).T, U2))

            e = LA.norm(U2-U, np.inf)
            U = np.copy(U2)
            temp += 1
        print(f"Policzono w {temp} kroków")
        self.temp: int = temp
        return U
    
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

    n: int = 10   #Liczba węzłów wewnętrzych dla x
    m: int = 15  #Liczba węzłów wewnętrzych dla y

    peaceman: Peaceman = Peaceman(f, lowerU, upperU, leftU, rightU)
    peaceman.setNM(n,m)
    peaceman.setInterval([x0,xn], [y0,yn])
    peaceman.addSolution(udok)
    peaceman.setError(10**-10, 100000)
    print(peaceman)

if __name__ == '__main__':
    main()
