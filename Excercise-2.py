from Siatki_adaptacyjne import Adaptive_net
from Poisson import Poisson
from Poisson_GaussSeid import GaussSeid
from Poisson_Jakobi import Jacobi
from Relaxation import Relaxation
from Peaceman import Peaceman
from numpy import exp
from numpy import linalg as LA
import numpy as np
import pandas as pd
import time

def main() -> None:
    #Równanie Różniczkowe cząstkowe rzędu II typu eliptycznego
    f = lambda x, y: (x**2 + y**2)+exp(x*y)
    #Warunki brzegowe
    lowerU = lambda x: 1
    upperU = lambda x: exp(x)
    leftU = lambda y: 1
    rightU = lambda y: exp(2*y)
    #Przedziały x i y
    x0 : float = 0
    xn : float = 2
    y0 : float = 0
    yn : float = 1
    #Rozwiązanie dokładne
    udok = lambda x,y: exp(x*y)

    n : int = 10  #Liczba węzłów wewnętrzych dla x
    m : int = 10  #Liczba węzłów wewnętrzych dla y

    poisson : GaussSeid = Adaptive_net(f, lowerU, upperU, leftU, rightU)
    poisson.setNM(n,m)
    poisson.setInterval([x0,xn], [y0,yn])
    poisson.addSolution(udok)
    poisson.setLabel(r"Równanie: $\frac{\partial^2\psi}{\partial x^2}+ \frac{\partial^2\psi}{\partial y^2} = (x^2+y^2)+e^{xy}$")
    poisson.setError(10**-15,10_000)
    poisson.addSolution(udok)
    print(poisson)
if __name__ == "__main__":
    main()