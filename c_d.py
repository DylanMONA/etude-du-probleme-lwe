import matplotlib.pyplot as plt
import random
import numpy as np

# Paramètres
q = 17
alpha = 0.5      #0.5 0.35 0.3 0.005
n_samples = 200
s_true = random.randint(0, q - 1)  # secret réel dans Z_q

#observer un pic a l'endroit ou le candidat est le bon pour lwe
#variance car une faible variance dans les erreurs indique que le bruit est concentre autour d'une valeur
#recentrage de la variance car pas pratique sinon on a 0 ou q quand c'est bien


# Distribution bruit gaussien modulo q
def DZq(alpha, q):
    return random.gauss(0, alpha * q) % q

# Produit scalaire 
def ps(a, s):
    return a * s

# Génération d'un échantillon LWE
def LWE(s, q, alpha):
    a = random.randint(0, q - 1)
    e = DZq(alpha, q)
    b = (a * s + e) % q
    return (a, b)

# Génération d'un échantillon uniforme
def uniform(q):
    a = random.randint(0, q - 1)
    b = random.randint(0, q - 1)
    return (a, b)

# Génération des données
def generateur(q, alpha, s_true, n_samples, use_LWE):
    data = []
    for _ in range(n_samples):
        if use_LWE:
            data.append(LWE(s_true, q, alpha))
        else:
            data.append(uniform(q))
    return data

# Tester les candidats s et calculer la variance des erreurs
def test_candidates(samples, q):
    variances = []
    for s_candidate in range(q):
        errors = [(b - a * s_candidate) % q for a, b in samples]
        # Ramener les erreurs dans l'intervalle [-q/2, q/2] pour une meilleure mesure
        centered_errors = [((e + q//2) % q) - q//2 for e in errors]
        variance = np.var(centered_errors)
        variances.append(variance)
    return variances

# Générer les deux types d'échantillons
samples_lwe = generateur(q, alpha, s_true, n_samples, use_LWE=True)
samples_uniform = generateur(q, alpha, s_true, n_samples, use_LWE=False)

# Calcul des variances pour chaque candidat s
variances_lwe = test_candidates(samples_lwe, q)
variances_uniform = test_candidates(samples_uniform, q)

# Affichage
plt.figure(figsize=(12, 6))
plt.plot(range(q), variances_lwe, label='Échantillons LWE', color='red')
plt.plot(range(q), variances_uniform, label='Échantillons Uniformes', color='blue')
plt.axvline(s_true, color='green', linestyle='--', label=f'Secret réel s = {s_true}')
plt.xlabel("Candidat s")
plt.ylabel("Variance des erreurs")
plt.title("Distinguer LWE vs Uniforme par la variance des erreurs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("CALC_DECI.png")


