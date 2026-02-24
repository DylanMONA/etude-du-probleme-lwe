
#Projet de Cryptologie — LWE et Chiffrement par Attributs

Dylan Mona — M1 MIC
Université Paris-Cité
Année 2024–2025

#Description du projet

Ce projet explore le problème Learning With Errors (LWE), pierre angulaire de la cryptographie post-quantique moderne, ainsi que plusieurs constructions cryptographiques basées sur les réseaux euclidiens.

Nous étudions :

Les versions calculatoire et décisionnelle du LWE et leur équivalence

Le chiffrement de Regev

Les outils fondamentaux liés aux réseaux (trappes, bases courtes, échantillonnage gaussien)

Le chiffrement par attributs 

Le protocole de Sergey Gorbunov, Vinod Vaikuntanathan et Hoeteck Wee (GVW)

# Structure des fichiers code
├── lwevsU.py        # Comparaison LWE vs uniforme
├── c_d.py           # Calculatoire => Décisionnel
├── d_c.py           # Décisionnel => Calculatoire
├── regev.py         # Implémentation du chiffrement de Regev
├── attribut.py      # Visualisation circuits
├── att.py           # Implémentation GVW

