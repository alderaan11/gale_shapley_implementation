from typing import Dict, List
from copy import deepcopy
import typer
from typing import Dict, List
from pathlib import Path
from utils import load_data_from_json
import random
from algorithms import gale_shapley_etudiant_optimal, gale_shapley_university_optimal, est_stable
import json
import datetime



app = typer.Typer()


def swing_improvement(
    unstable_matching: Dict[str, str],
    prefs_etus: Dict[str, List[str]],
    prefs_unis: Dict[str, List[str]],
    capacities: Dict[str, int],
    max_iterations: int = 30,
    verbose: bool = True
):

    match = deepcopy(unstable_matching)

    all_unis = set(match.values())
    fixed_cap = {uni: capacities.get(uni, 1) for uni in all_unis}

    uni_to_etu = {u: [] for u in fixed_cap}
    for etu, uni in match.items():
        uni_to_etu[uni].append(etu)

    def satisfaction(etu, uni):
        prefs = prefs_etus[etu]
        return 1 - (prefs.index(uni) / (len(prefs) - 1))

    def uni_prefers(uni, a, b):
        prefs = prefs_unis[uni]
        return prefs.index(a) < prefs.index(b)


    def log(message):
        if verbose:
            print(message)

    log("\n===== DÉBUT DE SWING =====\n")
    log(f"Matching initial : {match}\n")

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        log(f"\n--- ITERATION {iteration} ---")

        # Trier les étudiants du moins au plus satisfait
        sorted_etus = sorted(match.keys(), key=lambda e: satisfaction(e, match[e]))

        # On swing les 20% les moins satisfaits
        k = max(1, len(sorted_etus) // 5)
        worst_etus = sorted_etus[:k]

        log(f"Étudiants les moins satisfaits : {worst_etus}")

        for etu in worst_etus:
            current_uni = match[etu]
            log(f"\n>>> Tentative d'amélioration pour {etu} (actuellement à {current_uni})")

            for target_uni in prefs_etus[etu]:

                # ignorer si c’est pire que son assignation actuelle
                if prefs_etus[etu].index(target_uni) > prefs_etus[etu].index(current_uni):
                    continue

                log(f"  - Considère {target_uni}")

                # CAS 1 : place libre
                if len(uni_to_etu[target_uni]) < fixed_cap[target_uni]:

                    log(f"       Place disponible à {target_uni} → déplacement")
                    log(f"       {etu}: {current_uni} → {target_uni}")

                    uni_to_etu[current_uni].remove(etu)
                    uni_to_etu[target_uni].append(etu)
                    match[etu] = target_uni

                    improved = True
                    break

                # CAS 2 : pleine : essayer un swap
                current_students = uni_to_etu[target_uni]
                worst_current = max(
                    current_students,
                    key=lambda x: prefs_unis[target_uni].index(x)
                )

                log(f"       Université pleine. Étudiant le moins préféré : {worst_current}")

                if uni_prefers(target_uni, etu, worst_current):

                    log(f"       SWAP : {etu} prend la place de {worst_current} à {target_uni}")
                    log(f"       - {etu} : {current_uni} → {target_uni}")
                    log(f"       - {worst_current} : {target_uni} → {current_uni}")

                    # effectuer le swap
                    uni_to_etu[current_uni].remove(etu)
                    uni_to_etu[target_uni].append(etu)
                    match[etu] = target_uni

                    uni_to_etu[target_uni].remove(worst_current)
                    uni_to_etu[current_uni].append(worst_current)
                    match[worst_current] = current_uni

                    improved = True
                    break

                else:
                    log(f"      {target_uni} préfère garder {worst_current}")

            if improved:
                log("\nUne amélioration a été trouvée, on relance l'itération.")
                break

        if not improved:
            log("\nAucune amélioration trouvée cette itération.")

    typer.secho("\n--- Fin ---", fg=typer.colors.YELLOW)
    log(f"Matching final : {match}\n")

    return match


@app.command()
def main(

):
    k=1
    alpha=0.34
    beta=0.33
    gamme=0.33

    prefs_etudiants =  {
    "i1" : ['s2', 's1', 's3'],
    "i3" : ['s1', 's2', 's3'],
    "i2" : ['s1', 's2', 's3'],
    }
    prefs_uni =  {
    "s1" : ['i1', 'i3', 'i2'],
    "s3" : ['i2', 'i1', 'i3'],
    "s2" : ['i2', 'i1', 'i3'],
    }

    typer.secho("\n--- Aperçu des préférences générées ---", fg=typer.colors.CYAN)
    for s, prefs in list(prefs_etudiants.items())[:5]:
        print(f"{s} : {prefs}")
    for e, prefs in list(prefs_uni.items())[:5]:
        print(f"{e} : {prefs}")
    typer.secho("\n--- Exécution de Gale–Shapley (étudiant-optimal) ---", fg=typer.colors.GREEN)
    matching_etudiant_opt = gale_shapley_etudiant_optimal(
        prefs_etudiants, prefs_uni
    )
    print(matching_etudiant_opt)
    stable_e, _ = est_stable(matching_etudiant_opt, prefs_etudiants, prefs_uni)
    typer.echo(f"Stable (étudiant-optimal) : {stable_e}")
    

    typer.secho("\n--- Exécution de Gale–Shapley (université-optimal) ---", fg=typer.colors.GREEN)
    matching_universite_opt = gale_shapley_university_optimal(
        prefs_etudiants, prefs_uni
    )
    stable_u, _ = est_stable(matching_universite_opt, prefs_etudiants, prefs_uni)
    print(matching_universite_opt)
    typer.echo(f"Stable (université-optimal) : {stable_u}")
    
    unstable_matching = {'i1': 's2', 'i3': 's3', 'i2': 's1'}
    print("fake stabilty", est_stable(unstable_matching, prefs_etudiants, prefs_uni))



    typer.secho("\n--- Calcul des scores ---", fg=typer.colors.MAGENTA)
    from score import score_final

    score_etud_opt = score_final(
        unstable_matching,
        matching_etudiant_opt,
        matching_universite_opt,
        prefs_etudiants,
        prefs_uni,
        k=k,
        alpha=alpha,
        beta=beta,
        gamma=gamme
    )


    satisfaction_global_etudopt = score_etud_opt["score_final"] 
    print(satisfaction_global_etudopt)
    threshold = 1.1
    matching_swing = None


    if satisfaction_global_etudopt < threshold:
        print("in swine")
        typer.secho(
            f"\n--- Score étudiant-optimal < {threshold*100}% ({satisfaction_global_etudopt:.2f}). "
            "Application de l’algorithme SWING ---",
            fg=typer.colors.YELLOW
        )

        from swing import swing_improvement
        
        matching_swing = swing_improvement(
            unstable_matching,
            prefs_etudiants,
            prefs_uni,
            {"s1":1, "s2":1, "s3":1}
        )

        typer.secho("\n--- Nouveau score après SWING ---", fg=typer.colors.YELLOW)

        score_swing = score_final(
            matching_swing,
            matching_etudiant_opt,
            matching_universite_opt, 
            prefs_etudiants,
            prefs_uni,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamme
        )    
        typer.secho("\n--- Score après SWING ---", fg=typer.colors.BRIGHT_GREEN)

        typer.echo(score_swing)


    else:
        typer.secho(
            f"\n--- Score étudiant-optimal ≥ {threshold}% ({satisfaction_global_etudopt:.2f}). "
            "SWING non appliqué ---",
            fg=typer.colors.BLUE
        )






    typer.secho("\n--- Fin ---", fg=typer.colors.BRIGHT_GREEN)


    out_data = {
        "preferences": {
            "etudiants": prefs_etudiants,
            "etablissements": prefs_uni,
        },
        "matchings": {
            "unstable_matching": unstable_matching,
            "etudiant_optimal": matching_etudiant_opt,
            "universite_optimal": matching_universite_opt
        },
        "scores": {
            "student_optimal": score_etud_opt,
                            }
    }
    
    if matching_swing is not None:
        out_data["matchings"]["swing"] = matching_swing
        out_data["scores"]["swing"] = score_swing

    typer.echo("\nRECAP :")
    typer.echo(score_etud_opt)



if __name__ == "__main__":
    app()

    