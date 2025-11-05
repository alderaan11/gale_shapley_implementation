from stable_marriage import *

def main():
    print("=== Simulateur de l’algorithme du mariage stable ===")

    nb_etudiants = int(input("Entrez le nombre d’étudiants : "))
    nb_etablissements = int(input("Entrez le nombre d’établissements : "))
    capacite = int(input("Entrez la capacité par établissement : "))

    # Génération aléatoire des préférences
    prefs_etudiants, prefs_etablissements, capacites = generer_preferences(nb_etudiants, nb_etablissements, capacite)

    print("\n--- Préférences des étudiants ---")
    for e in list(prefs_etudiants.keys())[:5]:
        print(f"{e} : {prefs_etudiants[e]}")

    print("\n--- Préférences des établissements ---")
    for etab in list(prefs_etablissements.keys())[:5]:
        print(f"{etab} : {prefs_etablissements[etab]}")

    # Exécution de l’algorithme de Gale–Shapley
    appariements = gale_shapley(prefs_etudiants, prefs_etablissements, capacites)

    print("\n--- Résultats de l’appariement ---")
    for etab, etus in appariements.items():
        print(f"{etab} ← {etus}")

    # Calcul des niveaux de satisfaction
    moy_etu, moy_etab = calculer_satisfaction(appariements, prefs_etudiants, prefs_etablissements)
    stable, paire = est_stable(appariements, prefs_etudiants, prefs_etablissements)

    print("\n--- Statistiques finales ---")
    print(f"Satisfaction moyenne des étudiants : {moy_etu:.3f}")
    print(f"Satisfaction moyenne des établissements : {moy_etab:.3f}")
    print(f"Appariement stable : {'Oui' if stable else 'Non'}")
    if not stable:
        print(f"Paire bloquante : {paire}")

if __name__ == "__main__":
    main()
