import numpy as np
from numpy.random import default_rng
import random


rng = default_rng()

# Paramètres
q = 673 #12230087
n = 20
m = 100
alpha = 1e-12 # Réduction du bruit 1e-09 1e-11  1e12
 
def sample_secret(n):
    return rng.integers(0, q, size=n)

def sample_matrix(m, n):
    return rng.integers(0, q, size=(m, n))

def discrete_gaussian(stddev):
    e = [random.gauss(0,stddev) for _ in range(m)]
    
    return e 


def lwe_encrypt(A, s, e, message_bit):
    return (A @ s + e + (q // 2) * message_bit) % q

def and_gate(b0, b1): return b0 & b1

def or_gate(b0, b1): return b0 | b1

def eval_gate(gate_type, b0, b1):
    return and_gate(b0, b1) if gate_type == "AND" else or_gate(b0, b1)

class Gate:
    def __init__(self, gate_type, in0, in1, out):
        self.type = gate_type
        self.in0 = in0
        self.in1 = in1
        self.out = out

def make_example_circuit():
    gates = []
    for i in range(0, 8, 2):
        gates.append(Gate("AND", i, i+1, 8 + i//2))
    gates.append(Gate("OR", 8, 9, 12))
    gates.append(Gate("OR", 10, 11, 13))
    gates.append(Gate("AND", 12, 13, 14))
    return gates



import numpy as np

def sample_small_matrix(rows, cols, bound=500):
    
    return np.random.randint(-bound, bound + 1, size=(rows, cols))

def gadget_matrix(m, base=2):
    
    return np.diag([base ** i for i in range(m)])

def genbasis(m, n, q):
    if m <= n:
        raise ValueError(f"m doit être strictement supérieur à n. Reçu m={m}, n={n}")
    
    # 1. Choisir R avec petits coefficients dans [-B, B]
    B = int(np.ceil(np.log2(q)))
    R = np.random.randint(-B, B+1, size=(m - n, n))  # (m-n) x n

    # 2. Construire A = [ -R^T | I_n ]^T mod q → A de dimension (m x n)
    # A^T = [-R | I_{m-n}]  donc A = A_T.T = [[-R^T], [I_{m-n}]]  → m x n
    A_top = (-R) % q            
    A_bottom = np.eye(n, dtype=int) % q  

    A = np.vstack([A_top, A_bottom])  

    # 3. Construire S ∈ Z^{m x m}
    # S = [[q*I_n,     0     ],
    #      [ R   ,  I_{m-n} ]]
    top_left = q * np.eye(n, dtype=int)
    top_right = np.zeros((n, m - n), dtype=int)
    bottom_left = R
    bottom_right = np.eye(m - n, dtype=int)

    S = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])  
    
    return A % q, S



def gauss_modular_solve(A, b, q):
    
    A = np.array(A, dtype=int)
    b = np.array(b, dtype=int).flatten()

    n, m = A.shape

    # Construction de la matrice augmentée [
    Ab = np.hstack([A, b.reshape(-1, 1)]) % q

    rank = 0
    for col in range(m):
        pivot_row = None
        for r in range(rank, n):
            if Ab[r, col] % q != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue

        # Échange de ligne
        if pivot_row != rank:
            Ab[[rank, pivot_row]] = Ab[[pivot_row, rank]]

        # Normaliser la ligne pivot
        inv_pivot = pow(int(Ab[rank, col]), -1, q)  # inverse mod q
        Ab[rank] = (Ab[rank] * inv_pivot) % q

        # Élimination sur les autres lignes
        for r in range(n):
            if r != rank and Ab[r, col] != 0:
                factor = Ab[r, col]
                Ab[r] = (Ab[r] - factor * Ab[rank]) % q

        rank += 1
        if rank == n:
            break


    # Trouver solution particulière 
    x = np.zeros(m, dtype=int)
    # Reprendre les pivots pour assigner valeurs
    pivot_cols = []
    r = 0
    for c in range(m):
        if r < rank and Ab[r, c] == 1:
            x[c] = Ab[r, -1]
            pivot_cols.append(c)
            r += 1

    # Les variables libres restent à 0 

    return x % q

def sample_preimage_gaussian(A, S, y, q, s):
    A = np.array(A, dtype=int)
    S = np.array(S, dtype=int)
    y = np.array(y, dtype=int).flatten()

    m, n = A.shape

    # Résoudre A^T x = y mod q
    x0 = gauss_modular_solve(A.T, y, q)

    # Échantillonner z gaussien
    k = S.shape[1]
    z = np.random.normal(loc=0, scale=s, size=k)
    z = np.round(z).astype(int)

    x1 = S @ z
    x = (x0 + x1) % q

    return x.astype(int)

    



def solve_reencryption_matrix_with_trapdoor(A0, A1, Atgt, S0):
    
    m, n = A0.shape
    R1 = np.random.normal(loc=0, scale=alpha*q, size=(m, m)).round().astype(int) 
    R0 = np.zeros((m, m), dtype=int)
    # Cible partielle : Atgt - R1 * A1 mod q
    target = (Atgt - np.dot(R1, A1)) % q

    for i in range(m):
        # Résoudre R0[i] * A0 = target[i] mod q
        # Échantillonnage Gaussien avec trappe S0
        

        t = target[i]
        r = sample_preimage_gaussian(A0, S0, t, q, alpha*q)
        R0[i] = r

    R = np.hstack((R0, R1)) % q
    return R




def keygen(num_inputs):
    mpk = {}
    msk = {}
    for i in range(num_inputs):
        for b in [0, 1]:
            A, S = genbasis(m, n, q)  # génère la matrice A et sa trappe S
            mpk[(i, b)] = A
            msk[(i, b)] = S

    # Aout et Aout_fake peuvent être générés aléatoirement sans trappe
    Aout = sample_matrix(m, n)
    Aout_fake = sample_matrix(m, n)
    mpk['Aout'] = Aout
    msk['Aout'] = Aout
    msk['Aout_fake'] = Aout_fake
    return mpk, msk


def encrypt(attrs, message_bits, mpk):
    s = sample_secret(n)
    c = []
    for i, bit in enumerate(attrs):
        A = mpk[(i, bit)]
        ei = discrete_gaussian( alpha * q)
        ci = (A @ s + ei) % q
        c.append(ci)

    eout = discrete_gaussian( alpha * q)
    cout = (mpk['Aout'] @ s + eout + (q // 2) * message_bits) % q
    return c, cout, s, eout

def make_reenc_matrix(A0, A1, Aout):
    combined = np.hstack((A0, A1))
    try:
        R, _, _, _ = np.linalg.lstsq(combined.T, Aout.T, rcond=None)
        return np.rint(R.T).astype(int) % q  
    except:
        return sample_matrix(m, 2 * m)

def generate_skC(circuit, mpk, msk, num_inputs, arrete):
    skC = {}
    wire_A = {}
    wire_S = {}

    for i in range(num_inputs):
        for b in [0, 1]:
            wire_A[(i, b)] = mpk[(i, b)]
            wire_S[(i,b)]= msk[(i,b)]


    # Étape 1 : Charger les matrices A_i^{(b)} pour les arêtes > num_inputs
    for i in range(num_inputs , arrete ):  
        for b in [0, 1]:
            A, S = genbasis(m, n, q)
            wire_A[(i, b)] = A
            wire_S[(i, b)] = S

    # Étape 2 : Traiter chaque porte du circuit
    for gate in circuit:
        gate_out = gate.out
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                
                out_bit = eval_gate(gate.type, b0, b1)

                A0 = wire_A[(gate.in0, b0)]
                S0= wire_S[(gate.in0, b0)]
                A1 = wire_A[(gate.in1, b1)]
                
                

                # Étape 3 : Choisir la bonne sortie Aout
                is_last_gate = (gate_out == circuit[-1].out)
                if is_last_gate:
                    Aout = mpk['Aout'] if out_bit == 1 else sample_matrix(m, n)
                else:
                    
                    Aout=wire_A[(gate_out, out_bit)]
                    

                # Étape 4 : Calcul de la matrice de rechiffrement
               
                R = solve_reencryption_matrix_with_trapdoor(A0, A1, Aout, S0)  # ∈ Z_q^{m x 2m}
               
                # Stocker R dans la clé de déchiffrement
                skC[(gate_out, b0, b1)] = R
                

    return skC



def decrypt(c, cout, circuit, skC, attrs):
    
    wire_c = {i: c[i] for i in range(len(attrs))}  # c_i pour chaque entrée
    attr_ext = attrs[:]  # bits de circuit intermédiaires

    for gate in circuit:
        b0 = attr_ext[gate.in0]
        b1 = attr_ext[gate.in1]
        
        # Clé secrète pour cette combinaison
        R = skC[(gate.out, b0, b1)]

        # Concaténation des chiffrés d'entrée de la porte
        combined = np.concatenate([wire_c[gate.in0], wire_c[gate.in1]])  # shape: (2n,)

        # Application de la matrice de rechiffrement
        wire_c[gate.out] = (R @ combined) % q  # R est (n x 2n) → résultat (n,)

        # Calcul du bit logique de sortie
        attr_ext.append(eval_gate(gate.type, b0, b1))  # booléen 0 ou 1
        print(eval_gate(gate.type, b0, b1))
    # À la sortie du circuit, on compare cout avec c_{|C|}
    c_final = wire_c[circuit[-1].out]

    # Soustraction modulaire
    final = (cout - c_final) % q

    # Décodage tolérant au bruit :
    
    recovered_bits = ((final > q // 4) & (final < 3 * q // 4)).astype(int)

    return recovered_bits


# Exécution
num_inputs = 8
arrete = 15
message_bits = rng.integers(0, 2, size=m)
attributes = [1,1,0,0,0,0,1,1]    #[1,1,0,0,0,0,1,1]   rng.integers(0, 2, size=num_inputs).tolist()
print(attributes)
mpk, msk = keygen(num_inputs)
circuit = make_example_circuit()
skC = generate_skC(circuit, mpk, msk, num_inputs,arrete)
cipher, cout, s, eout = encrypt(attributes, message_bits, mpk)
decrypted = decrypt(cipher, cout, circuit, skC, attributes[:])

#print("Message original    :", message_bits.tolist())
#print("Message déchiffré :", decrypted.tolist())
print("Taux de réussite     :", np.mean(decrypted == message_bits) * 100, "%")

