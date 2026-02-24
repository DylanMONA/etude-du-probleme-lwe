import numpy as np
import random
from math import log2
import matplotlib.pyplot as plt

# Paramètres
q = 673   #12289  # Nombre premier
n = 50     # Dimension du secret
m = int(4 * (n + 1) * log2(q))  #  m ≥ 4(n+1)log2(q)
alpha = 1/4  # ]0, 1/(4m)[  test 4   200    400   400 00

# si trop proche de 1 on ne differencie rien donc peut pas dechiffrer
#si trop proche de 0 on resout un simple systeme mod q

# Distribution gaussienne centrée en 0
def DZaq(alpha, q):
    s = alpha * q
    return random.gauss(0, s)

# Uniforme sur Z_q
def Z(q):
    return random.randint(0, q - 1)

# Uniforme sur {0,1}
def U():
    return random.randint(0, 1)

# Génération de clés
def keygen():
    A = np.random.randint(0, q, size=(m, n))  # matrice A ∈ Z_q^{m x n}
    s = [Z(q) for _ in range(n)]              # vecteur secret s ∈ Z_q^n
    e = [DZaq(alpha, q) for _ in range(m)]    # bruit e ∈ R^m
    b = (np.dot(A, s) + e) % q                # b = As + e mod q
    return (A, b), s

# Chiffrement
def enc(cle_pub, M):
    A, b = cle_pub
    r = [U() for _ in range(m)]               # vecteur binaire 
    u = np.dot(np.transpose(r),A) % q                      # u = r^T A mod q
    v = (np.dot( np.transpose(r),b) + (q // 2) * M) % q     # v = r^T b + ⌊q/2⌋*M mod q
    return u, v

# Déchiffrement
def dec(cle_pv, chiffre):
    s = cle_pv
    u, v = chiffre
    res = (v - np.dot(np.transpose(u) , s)) % q
    #plus proche de 0 ou de q/2
    if (res < q // 4 or res > 3 * q // 4):
        return 0
    else :
        return 1

# Déchiffrement pour le graphe
def decG(cle_pv, chiffre):
    s = cle_pv
    u, v = chiffre
    res = (v - np.dot(np.transpose(u) , s)) % q
    return res


def pointgrap(nb_tests=1000):
    cas0 = []
    cas1 = []
    erreurs = 0
    cle_pub, cle_pv = keygen()

    for _ in range(nb_tests):
        for bit in [0, 1]:
            u, v = enc(cle_pub, bit)
            res2=dec(cle_pv,(u,v))
            res = decG(cle_pv, (u, v))
            if res2 != bit:
                erreurs += 1
                print("Erreur de déchiffrement")
            if bit == 0:
                cas0.append(res)
            else:
                cas1.append(res)
    print("Nombre d'erreurs de déchiffrement:", erreurs)
    print("Taux d'erreur:", erreurs / (2 * nb_tests))
    return cas0, cas1

# Main
def main():
    #test un cas 
    cle_pub,  cle_pv = keygen()
    message = 1
    chiffre = enc( cle_pub, message)
    claire = dec(cle_pv, chiffre)
    print("Message original:", message)
    print("Chiffre u:", chiffre[0], "\nChiffre v" , chiffre[1])
    print("Message déchiffré:", claire)
    
    

    #graphe
    rescas0, rescas1 = pointgrap()
    #print("Résultats de déchiffrement pour bit original = 0:", rescas0)
    #print("Résultats de déchiffrement pour bit original = 1:", rescas1) 
    plt.hist(rescas0, bins=50, alpha=0.6, label="Bit original = 0", color='blue',density=True)
    plt.hist(rescas1, bins=50, alpha=0.6, label="Bit original = 1", color='orange',density=True)
    plt.axvline(q // 4, color='black', linestyle='--', label='q/4')
    plt.axvline(3 * q // 4, color='black', linestyle='--', label='3q/4')
    plt.title("Histogramme des valeurs de déchiffrement avant arrondi")
    plt.xlabel("Résultat brut de v - <u, s> mod q")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.grid(True)
    plt.savefig("regev.png")
    plt.close()



if __name__ == "__main__":
    main()
