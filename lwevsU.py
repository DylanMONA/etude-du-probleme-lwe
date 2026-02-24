import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Paramètres
q = 97
n = 1  
alpha = 0.2   #0.2  0.5 0.05
n_samples = 500

# plus l'erreu est faible plus on distingue
#0.



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

def main():
    #secret
    s = [Z(q) for _ in range(n)]

    lwe_points = []
    uniform_points = []
    # Generation des points LWE et uniformes
    for _ in range(n_samples):
        a, b = DLWE(s, n, q, alpha)
        lwe_points.append((a[0], b))

        a_uniform = [Z(q) for _ in range(n)]
        b_uniform = Z(q)
        uniform_points.append((a_uniform[0], b_uniform))

    # Dessine le graphe
    plt.figure(figsize=(10, 5))
    plt.scatter(*zip(*lwe_points), color='red', alpha=0.6, label='LWE')
    plt.scatter(*zip(*uniform_points), color='blue', alpha=0.4, label='Uniforme')
    plt.title("Comparaison LWE vs Uniforme ")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.legend()
    plt.grid(True)
    plt.savefig("LvsU.png")
    plt.close()


if __name__ == "__main__":
    main()
