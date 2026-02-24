import random
from math import exp, pi


# Paramètres
q = 673   #12289  # le modulo Nombre premier
n = 5     # Dimension du secret
m = 10  # Nombre d'échantillons à générer
alpha = 1/40 # facteur d'erreur ]0, 1[
def main():
    
    
    
    #version matricielle
    s = [Z(q) for _ in range(n)]

    A=[0]*m
    b=[0]*m
    for i in range (m):
        (A[i], b[i]) = DLWE(s, n, q, alpha)

    for j in range (m):
        for i in range (n):
            print(f"{A[j][i]}s_{i}", end="")
            if(i!=n-1):
                print(" + ",end="")
        print(" = ", int(b[j]) ,"mod",q)




    return True






def DLWE(s, n, q, alpha): 
    a = [0]*n
    for i in range(n):
        a[i]=Z(q)
    e = DZaq(alpha, q)
    b = (ps(a, s) + e) % q
    return (a, b)


def ps(u, v):
    res=0
    for i in range(len(u)):
        res=res+u[i]*v[i]
    
    return res


def Z(q):
    return random.randint(0, q - 1)


def DZaq(alpha, q):
    s = alpha * q
    return random.gauss(0, s)


if __name__ == "__main__":
    main()
