from typing import Dict, List
from copy import deepcopy
from algorithms import est_stable, gale_shapley_etudiant_optimal, gale_shapley_university_optimal
from score import score_final
import typer
import json
from pathlib import Path
import datetime

app = typer.Typer()



def rank(agent: str, target: str, prefs: Dict[str, List[str]]) -> int:
    return prefs[agent].index(target)


def find_rotation(matching, prefs_etu, prefs_uni):
    
    R = []

    for e in prefs_etu:
        current_uni = matching[e]
        idx = prefs_etu[e].index(current_uni)

        
        if idx + 1 >= len(prefs_etu[e]):
            continue

        better_uni = prefs_etu[e][idx + 1]

        
        current_partner = None
        for e2, u2 in matching.items():
            if u2 == better_uni:
                current_partner = e2
                break

        
        if rank(better_uni, e, prefs_uni) < rank(better_uni, current_partner, prefs_uni):
            R.append((e, current_uni, better_uni))

    if not R:
        return None
    return R


def apply_rotation(matching, rotation):
    new_m = deepcopy(matching)

    for (e, old_u, new_u) in rotation:

        new_m[e] = new_u

        for e2 in new_m:
            if e2 != e and new_m[e2] == new_u:
                new_m[e2] = old_u
                break

    return new_m




def swing(
    stable_matching: Dict[str, str],
    prefs_etu: Dict[str, List[str]],
    prefs_uni: Dict[str, List[str]],
    score_func,
    max_iter: int = 50
):
   

    current = deepcopy(stable_matching)
    current_score = score_func(current)

    for _ in range(max_iter):

        rotation = find_rotation(current, prefs_etu, prefs_uni)
        if rotation is None:
            break

        candidate = apply_rotation(current, rotation)

        # stabilité obligatoire
        ok, _ = est_stable(candidate, prefs_etu, prefs_uni)
        if not ok:
            continue

        candidate_score = score_func(candidate)

        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score
        else:
            break

    return current, current_score



@app.command()
def main():
    k = 1
    alpha = 0.34
    beta = 0.33
    gamma = 0.33

    prefs_etudiants = {
        "i1": ['s2', 's1', 's3'],
        "i3": ['s1', 's2', 's3'],
        "i2": ['s1', 's2', 's3'],
    }
    prefs_uni = {
        "s1": ['i1', 'i3', 'i2'],
        "s3": ['i2', 'i1', 'i3'],
        "s2": ['i2', 'i1', 'i3'],
    }

    typer.secho("\n--- Gale–Shapley étudiant-optimal ---", fg=typer.colors.GREEN)
    matching_etu_opt = gale_shapley_etudiant_optimal(prefs_etudiants, prefs_uni)
    print(matching_etu_opt)
    print("Stable :", est_stable(matching_etu_opt, prefs_etudiants, prefs_uni)[0])

    typer.secho("\n--- Gale–Shapley université-optimal ---", fg=typer.colors.GREEN)
    matching_uni_opt = gale_shapley_university_optimal(prefs_etudiants, prefs_uni)
    print(matching_uni_opt)
    print("Stable :", est_stable(matching_uni_opt, prefs_etudiants, prefs_uni)[0])

    unstable_matching = {'i1': 's2', 'i3': 's3', 'i2': 's1'}

    score_unstable = score_final(
        unstable_matching, matching_etu_opt, matching_uni_opt,
        prefs_etudiants, prefs_uni,
        k=k, alpha=alpha, beta=beta, gamma=gamma
    )

    typer.secho("\n--- Score initial ---", fg=typer.colors.MAGENTA)
    print(score_unstable)

    score_func = lambda m: score_final(
        m, matching_etu_opt, matching_uni_opt,
        prefs_etudiants, prefs_uni,
        k=k, alpha=alpha, beta=beta, gamma=gamma
    )["score_final"]

    
    typer.secho("\n--- Application de SWING ---", fg=typer.colors.YELLOW)
    matching_swing, score_swing = swing(
        matching_etu_opt,
        prefs_etudiants,
        prefs_uni,
        score_func
    )

    typer.secho("\n--- Matching SWING ---", fg=typer.colors.GREEN)
    print(matching_swing)
    print("Stable :", est_stable(matching_swing, prefs_etudiants, prefs_uni)[0])
    print("Score :", score_swing)

    typer.secho("\n--- FIN ---", fg=typer.colors.BRIGHT_GREEN)


if __name__ == "__main__":
    app()
