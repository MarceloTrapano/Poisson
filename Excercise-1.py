from Adaptive_grid import Adaptive_grid
from FDM_method import Poisson
from Poisson_GaussSeid import GaussSeid
from Poisson_Jakobi import Jacobi
from Relaxation import Relaxation
from Peaceman import Peaceman
from numpy import linalg as LA
import numpy as np
import pandas as pd
import time

def main() -> None:
    #Równanie Różniczkowe cząstkowe rzędu II typu eliptycznego
    f = lambda x, y: 0
    #Warunki brzegowe
    lowerU = lambda x: 0
    upperU = lambda x: x
    leftU = lambda y: 0
    rightU = lambda y: y
    #Przedziały x i y
    x0 : float = 0
    xn : float = 1
    y0 : float = 0
    yn : float = 1
    #Rozwiązanie dokładne
    udok = lambda x,y: x*y

    n : int = 5   #Liczba węzłów wewnętrzych dla x
    m : int = 5  #Liczba węzłów wewnętrzych dla y

    poisson = Poisson(f, lowerU, upperU, leftU, rightU)
    poisson.setNM(n,m)
    poisson.setLabel(r"Równanie: $\frac{\partial^2\psi}{\partial x^2}+ \frac{\partial^2\psi}{\partial y^2} = 0$")
    poisson.setInterval([x0,xn], [y0,yn])
    poisson.addSolution(udok)

    error = []
    czas = []
    for n in [5, 10, 20, 30, 40, 50]:
        X, Y = np.meshgrid(np.linspace(x0, xn, n+2),np.linspace(x0, xn, n+2))
        poisson.setNM(n,n)
        start_time = time.time()
        Un = poisson.solve()
        
        end_time = time.time()
        czas.append(end_time - start_time)
        error.append(poisson.show())
    d = {'n' : [5, 10, 20, 30, 40, 50], 'm' : [5, 10, 20, 30, 40, 50], 'Błąd' : error, 'Czas' : czas}
    array = pd.DataFrame(d)
    array.to_csv('out.csv', index=False)

if __name__ == "__main__":
    main()