from Siatki_adaptacyjne import Adaptive_net
from Poisson import Poisson
from Poisson_GaussSeid import GaussSeid
from Poisson_Jakobi import Jacobi
from Relaxation import Relaxation
from Peaceman import Peaceman
from numpy import log
import numpy as np
import pandas as pd
import time
from numpy import linalg as LA
def main() -> None:
    #Równanie Różniczkowe cząstkowe rzędu II typu eliptycznego
    f = lambda x, y: x/y+y/x
    #Warunki brzegowe
    lowerU = lambda x: x*log(x)
    upperU = lambda x: x*log(4*x**2)
    leftU = lambda y: y*log(y)
    rightU = lambda y: 2*y*log(2*y)
    #Przedziały x i y
    x0 : float = 1
    xn : float = 2
    y0 : float = 1
    yn : float = 2
    #Rozwiązanie dokładne
    udok = lambda x,y: x*y*log(x*y)

    n : int = 5   #Liczba węzłów wewnętrzych dla x
    m : int = 5  #Liczba węzłów wewnętrzych dla y

    poisson : Jacobi = Adaptive_net(f, lowerU, upperU, leftU, rightU)
    poisson.setNM(n,m)
    poisson.setInterval([x0,xn], [y0,yn])
    poisson.setLabel(r"Równanie: $\frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} = \frac{x}{y} + \frac{y}{x}$")
    poisson.addSolution(udok)
    poisson.setError(10**-10, 10_000)
    error = []
    czas = []
    iter = []
    r = [2, 5, 10, 15, 20, 25]
    for n in r:
        X, Y = np.meshgrid(np.linspace(x0, xn, 2*n+3),np.linspace(x0, xn, 2*n+3))
        poisson.setNM(n,n)
        start_time = time.time()
        Un = poisson.solve()
        
        end_time = time.time()
        czas.append(end_time - start_time)
        iter.append(poisson.temp)
        error.append(LA.norm(udok(X,Y)-np.transpose(Un), np.inf))
    d = {'n' : r, 'm' : r, 'Błąd' : error, 'Iteracje' : iter, 'Czas' : czas}
    array = pd.DataFrame(d)
    array.to_csv('out.csv', index=False)

if __name__ == "__main__":
    main()