import json
import random
from pathlib import Path
from typing import Dict, List
import typer
from load_data import load_data_from_json
from datetime import datetime

app = typer.Typer()

# ------------------------------------------------------------
#  ALGOS D'APPARIEMENT
# ------------------------------------------------------------

def gale_shapley_student_optimal(prefs_students: Dict[str, List[str]],
                                 prefs_unis: Dict[str, List[str]],
                                 capacities: Dict[str, int]) -> Dict[str, List[str]]:
    """Gale–Shapley où les étudiants proposent (étudiant-optimal)."""
    free = list(prefs_students.keys())
    next_proposal = {s: 0 for s in prefs_students}
    matching = {u: [] for u in prefs_unis}

    while free:
        stu = free.pop(0)
        prefs = prefs_students[stu]

        if next_proposal[stu] >= len(prefs):
            continue

        uni = prefs[next_proposal[stu]]
        next_proposal[stu] += 1

        # Si l'établissement n'est pas plein, accepte
        if len(matching[uni]) < capacities[uni]:
            matching[uni].append(stu)
        else:
            ranking = prefs_unis[uni]
            # le pire actuellement accepté
            worst = max(matching[uni], key=lambda x: ranking.index(x))

            if ranking.index(stu) < ranking.index(worst):
                matching[uni].remove(worst)
                matching[uni].append(stu)
                free.append(worst)
            else:
                free.append(stu)
    return matching


def gale_shapley_university_optimal(prefs_students: Dict[str, List[str]],
                                    prefs_unis: Dict[str, List[str]],
                                    capacities: Dict[str, int]) -> Dict[str, List[str]]:
    """
    Version approximative où les établissements proposent.
    On ne s'en sert ici que pour comparaison, pas pour les rotations.
    """
    free = list(prefs_unis.keys())
    next_proposal = {u: 0 for u in prefs_unis}
    matching = {u: [] for u in prefs_unis}
    student_assignment = {s: None for s in prefs_students}

    while free:
        uni = free.pop(0)
        if next_proposal[uni] >= len(prefs_unis[uni]):
            continue

        stu = prefs_unis[uni][next_proposal[uni]]
        next_proposal[uni] += 1

        current = student_assignment[stu]

        if current is None:
            student_assignment[stu] = uni
            matching[uni].append(stu)
        else:
            ranking = prefs_students[stu]
            if ranking.index(uni) < ranking.index(current):
                matching[current].remove(stu)
                matching[uni].append(stu)
                student_assignment[stu] = uni
                # on remet l'ancienne université en file si elle a potentiellement des propositions à faire
                free.append(current)
            else:
                free.append(uni)

    return matching


# ------------------------------------------------------------
#  STABILITÉ
# ------------------------------------------------------------

def est_stable(matching: Dict[str, List[str]],
               prefs_students: Dict[str, List[str]],
               prefs_unis: Dict[str, List[str]]):
    """
    Vérifie la stabilité classique : pas de paire bloquante (étudiant, université).
    """
    # affectation étu -> uni
    assigned = {}
    for uni, stus in matching.items():
        for s in stus:
            assigned[s] = uni

    for stu, prefs in prefs_students.items():
        assigned_uni = assigned.get(stu)

        for uni in prefs:
            if uni == assigned_uni:
                break

            assigned_list = matching[uni]
            ranking = prefs_unis[uni]

            if not assigned_list:
                # université vide → clairement une paire bloquante potentielle
                return False, (stu, uni)

            worst = max(assigned_list, key=lambda x: ranking.index(x))

            if ranking.index(stu) < ranking.index(worst):
                return False, (stu, uni)

    return True, None


# ------------------------------------------------------------
#  SATISFACTION DE BASE
# ------------------------------------------------------------

def satisfaction_moyenne(matching: Dict[str, List[str]],
                         prefs_students: Dict[str, List[str]],
                         prefs_unis: Dict[str, List[str]]):
    """
    Satisfaction moyenne simple côté étudiants et côté établissements.
    (déjà proche de ce que tu avais)
    """
    sat_students = []
    sat_unis = []

    # Etudiants
    for stu, prefs in prefs_students.items():
        assigned = None
        for uni, stus in matching.items():
            if stu in stus:
                assigned = uni
                break
        if assigned is None:
            continue
        rank = prefs.index(assigned)
        sat_students.append((len(prefs) - rank) / len(prefs))

    # Etablissements
    for uni, prefs in prefs_unis.items():
        for stu in matching[uni]:
            rank = prefs.index(stu)
            sat_unis.append((len(prefs) - rank) / len(prefs))

    moy_etu = sum(sat_students) / len(sat_students) if sat_students else 0
    moy_uni = sum(sat_unis) / len(sat_unis) if sat_unis else 0
    return moy_etu, moy_uni


# ------------------------------------------------------------
#  COMPOSANTE TOP-k
# ------------------------------------------------------------

def satisfaction_top_k_students(matching, prefs_students, k: int = 3) -> float:
    scores = []
    for stu, prefs in prefs_students.items():
        assigned = None
        for uni, stus in matching.items():
            if stu in stus:
                assigned = uni
                break
        if assigned is None:
            scores.append(0)
            continue
        rank = prefs.index(assigned)
        if rank < k:
            scores.append(1 - (rank / k))
        else:
            scores.append(0)
    return sum(scores) / len(scores) if scores else 0


def satisfaction_top_k_unis(matching, prefs_unis, k: int = 3) -> float:
    scores = []
    for uni, prefs in prefs_unis.items():
        if not matching[uni]:
            continue
        for stu in matching[uni]:
            rank = prefs.index(stu)
            if rank < k:
                scores.append(1 - (rank / k))
            else:
                scores.append(0)
    return sum(scores) / len(scores) if scores else 0


# ------------------------------------------------------------
#  COMPOSANTE FRUSTRATION / QUASI-STABILITÉ
# ------------------------------------------------------------

def quasi_stability(matching, prefs_students, prefs_unis) -> float:
    """
    Mesure une "distance à l'instabilité" basée sur les paires quasi-bloquantes.
    Retourne un score entre 0 et 1 (1 = parfaitement stable).
    """
    total_frustration = 0
    count = 0

    # affectation étu -> uni
    assigned = {}
    for uni, stus in matching.items():
        for s in stus:
            assigned[s] = uni

    for stu, prefs in prefs_students.items():
        assigned_uni = assigned.get(stu)
        if assigned_uni is None:
            continue

        for uni in prefs:
            if uni == assigned_uni:
                break

            assigned_list = matching[uni]
            if not assigned_list:
                continue

            ranking = prefs_unis[uni]
            worst = max(assigned_list, key=lambda x: ranking.index(x))

            if ranking.index(stu) < ranking.index(worst):
                # Frustration côté étudiant
                fi = prefs.index(assigned_uni) - prefs.index(uni)
                # Frustration côté université
                fu = ranking.index(worst) - ranking.index(stu)
                total_frustration += max(0, (fi + fu) / 2)
                count += 1

    if count == 0:
        return 1.0  # parfaitement stable

    # normalisation simple
    max_frustration = count * 10.0
    return max(0.0, 1 - (total_frustration / max_frustration))

def plot_simplex_heatmap(X, Y, Z, title="Score simplexe"):
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(X, Y, c=Z, cmap="viridis", s=120, edgecolors="black")
    plt.colorbar(scatter)
    
    # tracer les frontières du triangle
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3)/2, 0]
    plt.plot(triangle_x, triangle_y, color="black")
    
    plt.title(title)
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_barycentric_surface(A, B, Z, title="Barycentric 3D Surface"):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(A, B, Z, cmap="viridis", edgecolor="none")
    
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("score")
    ax.set_title(title)
    plt.show()

# ------------------------------------------------------------
#  SCORE GLOBAL αA + βB + γC
# ------------------------------------------------------------
import numpy as np

def simplex_to_cartesian(alpha, beta, gamma):
    # alpha, beta, gamma sommation = 1
    # on utilise le simplex standard (triangle équilatéral)
    x = beta + gamma / 2
    y = (np.sqrt(3) / 2) * gamma
    return x, y

def score_global(matching,
                 prefs_students,
                 prefs_unis,
                 alpha: float = 0.4,
                 beta: float = 0.2,
                 gamma: float = 0.4,
                 k: int = 3):
    """
    Calcule :
    - S_etu : score global côté étudiants
    - S_uni : score global côté établissements
    - S_global : moyenne des deux
    """

    # A : top-k
    A_etu = satisfaction_top_k_students(matching, prefs_students, k)
    A_uni = satisfaction_top_k_unis(matching, prefs_unis, k)

    # B : satisfaction croisée moyenne
    B_etu, B_uni = satisfaction_moyenne(matching, prefs_students, prefs_unis)

    # C : frustration / quasi-stabilité (commune aux deux)
    C = quasi_stability(matching, prefs_students, prefs_unis)

    S_etu = alpha * A_etu + beta * B_etu + gamma * C
    S_uni = alpha * A_uni + beta * B_uni + gamma * C
    S_global = 0.5 * (S_etu + S_uni)

    return S_etu, S_uni, S_global


# ------------------------------------------------------------
#  ROTATIONS LOCALES POUR RÉDUIRE LE DELTA
# ------------------------------------------------------------

def equilibrer_satisfaction(matching,
                            prefs_students,
                            prefs_unis,
                            capacities,
                            alpha=0.4,
                            beta=0.4,
                            gamma=0.2,
                            seuil=0.1,
                            max_iterations=50):
    """
    Si la différence entre le score global étudiants et établissements > seuil,
    tente des "rotations locales" (swaps entre deux étudiants) pour réduire le delta,
    tout en gardant la stabilité.
    """

    for _ in range(max_iterations):
        S_etu, S_uni, S_global = score_global(
            matching, prefs_students, prefs_unis, alpha, beta, gamma
        )

        if abs(S_etu - S_uni) <= seuil:
            return matching, S_etu, S_uni, S_global, "OK (équilibré)"

        improved = False

        # Essayer toutes les paires d'universités
        unis = list(matching.keys())
        for i in range(len(unis)):
            for j in range(i + 1, len(unis)):
                u1, u2 = unis[i], unis[j]
                etus1, etus2 = matching[u1], matching[u2]

                # Essayer des swaps entre étudiants
                for e1 in etus1:
                    for e2 in etus2:
                        new_matching = {u: list(stus) for u, stus in matching.items()}
                        new_matching[u1].remove(e1)
                        new_matching[u1].append(e2)
                        new_matching[u2].remove(e2)
                        new_matching[u2].append(e1)

                        # Vérifier la stabilité
                        stable, _ = est_stable(new_matching, prefs_students, prefs_unis)
                        if not stable:
                            continue

                        # Calculer les nouveaux scores
                        S_etu2, S_uni2, S_global2 = score_global(
                            new_matching, prefs_students, prefs_unis, alpha, beta, gamma
                        )

                        # On garde uniquement si on réduit le delta
                        if abs(S_etu2 - S_uni2) < abs(S_etu - S_uni):
                            matching = new_matching
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            return matching, S_etu, S_uni, S_global, "Bloqué (pas d'amélioration)"

    # Si on sort par max_iterations
    S_etu, S_uni, S_global = score_global(
        matching, prefs_students, prefs_unis, alpha, beta, gamma
    )
    return matching, S_etu, S_uni, S_global, "Fin des itérations"

import csv

def experimenter_parametres(
    matching,
    prefs_students,
    prefs_unis,
    pas: float = 0.1,
    k: int = 3,
    output_csv: Path = None
):
    """
    Explore les effets de alpha, beta, gamma sur les scores.

    Si output_csv est fourni → enregistre un fichier CSV.
    """

    resultats = []

    # boucle triple si tu veux la contrainte alpha+beta+gamma=1
    for alpha in [round(i * pas, 2) for i in range(int(1/pas)+1)]:
        for beta in [round(i * pas, 2) for i in range(int((1-alpha)/pas)+1)]:
            gamma = round(1 - alpha - beta, 2)
            if gamma < 0:
                continue

            S_etu, S_uni, S_global = score_global(
                matching,
                prefs_students,
                prefs_unis,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                k=k
            )

            resultats.append({
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "score_etudiants": S_etu,
                "score_universites": S_uni,
                "score_global": S_global
            })

    # sauvegarde CSV
    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["alpha", "beta", "gamma", "score_etudiants", "score_universites", "score_global"]
            )
            writer.writeheader()
            writer.writerows(resultats)

    return resultats
def grid_params(pas=0.1):
    params = []
    for alpha in [round(i * pas, 2) for i in range(int(1/pas)+1)]:
        for beta in [round(i * pas, 2) for i in range(int((1-alpha)/pas)+1)]:
            gamma = round(1 - alpha - beta, 2)
            if gamma >= 0:
                params.append((alpha, beta, gamma))
    return params


def compute_param_grid_scores_full(matching, prefs_students, prefs_unis, pas=0.1, k=3):
    grid = grid_params(pas)
    
    X = []
    Y = []
    A = []  # alpha
    B = []  # beta
    G = []  # gamma
    
    S_etu = []
    S_uni = []
    S_glob = []

    for alpha, beta, gamma in grid:
        se, su, sg = score_global(matching, prefs_students, prefs_unis,
                                  alpha=alpha, beta=beta, gamma=gamma, k=k)

        # Transformer en coordonnées 2D du simplex
        x, y = simplex_to_cartesian(alpha, beta, gamma)
        
        X.append(x)
        Y.append(y)
        A.append(alpha)
        B.append(beta)
        G.append(gamma)
        S_etu.append(se)
        S_uni.append(su)
        S_glob.append(sg)

    return X, Y, A, B, G, S_etu, S_uni, S_glob



import matplotlib.pyplot as plt
import numpy as np

def plot_heatmaps(X, Y, Z, title="Heatmap score"):
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(X, Y, c=Z, cmap="viridis")
    plt.colorbar(scatter)
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title(title)
    plt.grid(True)
    plt.show()


from mpl_toolkits.mplot3d import Axes3D

def plot_surface_3d(X, Y, Z, title="Surface 3D score"):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("score")
    ax.set_title(title)

    plt.show()

# ------------------------------------------------------------
#  MAIN CLI
# ------------------------------------------------------------

@app.command()
def main(
    etudiants_json: Path = typer.Argument(..., help="Chemin du fichier JSON contenant les noms des étudiants."),
    etablissements_json: Path = typer.Argument(..., help="Chemin du fichier JSON contenant les noms des établissements."),
    output_dir: Path = typer.Option("./experiments_final", help="Dossier de sortie pour sauvegarder les préférences.")
):
    etudiants_data = load_data_from_json(etudiants_json)
    etudiants = etudiants_data["nom"]

    universite_data = load_data_from_json(etablissements_json)
    universites = [u["nom"] for u in universite_data["etablissements_superieurs_francais_complet"]]

    nb_etudiants = typer.prompt(f"Nombre d'étudiants à utiliser (max {len(etudiants)})", type=int)
    nb_uni = typer.prompt(f"Nombre d'établissements à utiliser (max {len(universites)})", type=int)
    capacite = typer.prompt("Capacité de chaque établissement", type=int)

    selec_etu = random.sample(etudiants, nb_etudiants)
    selec_uni = random.sample(universites, nb_uni)

    # Génération des préférences aléatoires
    prefs_etudiants = {
        etu: random.sample(selec_uni, len(selec_uni))
        for etu in selec_etu
    }
    prefs_uni = {
        uni: random.sample(selec_etu, len(selec_etu))
        for uni in selec_uni
    }

    capacites = {uni: capacite for uni in selec_uni}

    typer.secho("\n--- Aperçu des préférences générées ---", fg=typer.colors.CYAN)
    for s, prefs in list(prefs_etudiants.items())[:5]:
        print(f"{s} : {prefs}")
    for e, prefs in list(prefs_uni.items())[:5]:
        print(f"{e} : {prefs}")

    # ----------------- Matching étudiant-optimal -----------------
    typer.secho("\n=== Gale–Shapley étudiant-optimal ===", fg=typer.colors.GREEN)
    match_etu = gale_shapley_student_optimal(prefs_etudiants, prefs_uni, capacites)
    stable1, pb1 = est_stable(match_etu, prefs_etudiants, prefs_uni)
    S_etu1, S_uni1, S_global1 = score_global(match_etu, prefs_etudiants, prefs_uni)

    print("Stable :", stable1)
    print(f"Score global étudiants : {S_etu1:.3f}")
    print(f"Score global établissements : {S_uni1:.3f}")
    print(f"Score global moyen : {S_global1:.3f}")

    # Si >10% de différence, tenter équilibrage par rotations
    if abs(S_etu1 - S_uni1) > 0.10:
        typer.secho("\nDelta > 10% → tentative d'équilibrage par rotations locales...", fg=typer.colors.YELLOW)
        match_equil, S_etu2, S_uni2, S_global2, status = equilibrer_satisfaction(
            match_etu, prefs_etudiants, prefs_uni, capacites
        )
        typer.secho(f"Statut équilibrage : {status}", fg=typer.colors.CYAN)
        print(f"Après équilibrage → Score étudiants : {S_etu2:.3f}, établissements : {S_uni2:.3f}, global : {S_global2:.3f}")
        match_etu_final = match_equil
    else:
        match_etu_final = match_etu
    typer.secho("\n=== Variation des paramètres alpha, beta, gamma ===", fg=typer.colors.CYAN)
    typer.secho("\n=== Visualisation des effets des paramètres ===", fg=typer.colors.CYAN)

    X, Y, A, B, G, S_etu, S_uni, S_glob = compute_param_grid_scores_full(
        match_etu_final,
        prefs_etudiants,
        prefs_uni,
        pas=0.1,
        k=3
    )

    plot_simplex_heatmap(X, Y, S_glob, "Score global (simplex α-β-γ)")
    plot_simplex_heatmap(X, Y, S_etu, "Score étudiants (simplex)")
    plot_simplex_heatmap(X, Y, S_uni, "Score universités (simplex)")

    plot_barycentric_surface(A, B, S_glob, "Score global barycentrique 3D")

    # ----------------- Matching établissement-optimal (optionnel) -----------------
    typer.secho("\n=== Gale–Shapley établissement-optimal (comparaison) ===", fg=typer.colors.GREEN)
    match_uni = gale_shapley_university_optimal(prefs_etudiants, prefs_uni, capacites)
    stable2, pb2 = est_stable(match_uni, prefs_etudiants, prefs_uni)
    S_etu_u, S_uni_u, S_global_u = score_global(match_uni, prefs_etudiants, prefs_uni)

    print("Stable :", stable2)
    print(f"Score global étudiants : {S_etu_u:.3f}")
    print(f"Score global établissements : {S_uni_u:.3f}")
    print(f"Score global moyen : {S_global_u:.3f}")

    # ----------------- Sauvegarde -----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{output_dir}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "preference_et_resultats.json"

    out_data = {
        "etudiants": prefs_etudiants,
        "etablissements": prefs_uni,
        "capacites": capacites,
        "matching_student_optimal": match_etu_final,
        "matching_university_optimal": match_uni,
        "scores": {
            "student_optimal_initial": {
                "etu": S_etu1,
                "uni": S_uni1,
                "global": S_global1
            },
            "student_optimal_equilibre": {
                "etu": S_etu2 if abs(S_etu1 - S_uni1) > 0.10 else S_etu1,
                "uni": S_uni2 if abs(S_etu1 - S_uni1) > 0.10 else S_uni1,
                "global": S_global2 if abs(S_etu1 - S_uni1) > 0.10 else S_global1
            },
            "university_optimal": {
                "etu": S_etu_u,
                "uni": S_uni_u,
                "global": S_global_u
            }
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    typer.secho(f"\nRésultats sauvegardés dans : {output_path}", fg=typer.colors.GREEN)


if __name__ == '__main__':
    app()
