import numpy as np
import matplotlib.pyplot as plt

# Paramètres
n = 5           # dimension du secret
q = 17          # module
num_samples = 30
alpha= 0.005  # 0.15 0.1 0.05 0.005
sigma = alpha*q    # bruit gaussien



# Générer des échantillons LWE
def generate_lwe_sample(s, q, sigma):
    a = np.random.randint(0, q, size=n)
    e = np.random.normal(0, sigma)
    b = (np.dot(a, s) + e) % q
    return a, b



# Oracle de décision : renvoie vrai si l'entrée semble être un échantillon LWE
def oracle_decisional(a, b):
    
    e = (b - np.dot(a, s)) % q
    
    return abs((e + q // 2) % q - q // 2) < 3 * sigma

# Générer le secret aléatoire
s = np.random.randint(0, q, size=n)

#echantillon
samples = [generate_lwe_sample(s, q, sigma) for _ in range(num_samples)]

# Reconstitution du secret
recovered_s = np.zeros(n, dtype=int)

# Pour visualisation : probabilité d'acceptation pour chaque hypothèse s_i^*
prob_maps = []

for i in range(n):
    probs = []
    for s_i_star in range(q):
        count_LWE = 0
        trials = 10
        for _ in range(trials):
            a, b = samples[np.random.randint(0, num_samples)]
            u = np.random.randint(0, q)
            a_prime = (a + u * np.eye(1, n, i, dtype=int).flatten()) % q
            b_prime = (b + u * s_i_star) % q
            if oracle_decisional(a_prime, b_prime):
                count_LWE += 1
        probs.append(count_LWE / trials)
    recovered_s[i] = np.argmax(probs)
    prob_maps.append(probs)

# Affichage des résultats
fig, axes = plt.subplots(1, n, figsize=(15, 3))

fig.suptitle(f'secret :{s}')

for i in range(n):
    axes[i].bar(range(q), prob_maps[i])
    axes[i].set_title(f'Coordonnée {i}\n(s_i = {s[i]}, récupéré = {recovered_s[i]})')
    axes[i].set_xlabel('Hypothèse s_i^*')
    axes[i].set_ylabel('Probabilité LWE')

plt.tight_layout()
plt.savefig("d_c.png")
plt.close()
