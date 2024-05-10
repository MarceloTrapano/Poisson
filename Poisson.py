import numpy as np
from numpy import exp
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
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

    n : int = 100   #Liczba węzłów wewnętrzych dla x
    m : int = 150  #Liczba węzłów wewnętrzych dla y
    #Kroki dla x i y
    h : float = (xn-x0)/(n+1)
    k : float = (yn-y0)/(m+1)
    
    x : np.ndarray = np.linspace(x0, xn, n+2)
    y : np.ndarray = np.linspace(y0, yn, m+2)
    
    x1 : np.ndarray = x[1:n+1]
    y1 : np.ndarray = y[1:m+1]

    T : np.ndarray = np.diag(-2*(1/h**2+1/k**2)*np.ones(n)) + np.diag(1/h**2*np.ones(n-1),-1) + np.diag(1/h**2*np.ones(n-1),1)
    B : np.ndarray = np.diag(1/k**2*np.ones(n))
    A : np.ndarray = np.kron(np.diag(np.ones(m)),T) + np.kron(np.diag(np.ones(m-1),-1),B) + np.kron(np.diag(np.ones(m-1),1),B)
    
    F : list = []
    for j in range(m):
        for i in range(n):
            values : float = f(x1[i],y1[j])
            if j == 0:
                values -= lowerU(x1[i])/k**2
            elif j == m-1:
                values -= upperU(x1[i])/k**2
            if i == 0:
                values -= leftU(y1[j])/h**2
            elif i == n-1:
                values -= rightU(y1[j])/h**2
            F.append(values)
    F : np.ndarray = np.array(F)
    U : np.ndarray = LA.solve(A,F)
    U = U.reshape([m,n])
    U = np.flipud(U)
    print(U)
    X, Y = np.meshgrid(x,y)

    lowBoundX : list[float] = list(map(lowerU,x))
    upBoundX : list[float] = list(map(upperU,x))
    lowBoundY : list[float] = list(map(leftU,y))
    upBoundY : list[float] = list(map(rightU,y))
    lowBoundX = np.array(lowBoundX)
    upBoundX = np.array(upBoundX)
    lowBoundY = np.array(lowBoundY)
    upBoundY = np.array(upBoundY)

    U = np.vstack((U,lowBoundX[1:len(x)-1]))
    U = np.vstack((upBoundX[1:len(x)-1],U))

    U = np.hstack((np.flipud(lowBoundY.reshape([m+2,1])),U))
    U = np.hstack((U,np.flipud(upBoundY.reshape([m+2,1]))))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(X,Y,U)
    plt.show()
    
if __name__ == "__main__":
    main()