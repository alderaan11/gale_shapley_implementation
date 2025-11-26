from typing import Dict, List
from pathlib import Path
import typer
from utils import load_data_from_json
import random
from algorithms import gale_shapley_etudiant_optimal, gale_shapley_university_optimal, est_stable
import json
import datetime


app = typer.Typer()


@app.command()
def main(
    etudiants_json: Path = typer.Argument(..., help="Chemin du fichier JSON contenant les noms des étudiants."),
    etablissements_json: Path = typer.Argument(..., help="Chemin du fichier JSON contenant les noms des établissements."),
    output_dir: Path = typer.Option("./results_swing", help="Dossier de sortie pour sauvegarder les préférences.")
):
    k=1
    alpha=0.34
    beta=0.33
    gamme=0.33

    etudiants_data = load_data_from_json(etudiants_json)
    etudiants = etudiants_data["nom"]

    universite_data = load_data_from_json(etablissements_json)
    universites = [u["nom"] for u in universite_data["etablissements_superieurs_francais_complet"]]

    nb_etudiants = 3
    nb_uni = 3
    capacite = 1

    selec_etu = random.sample(etudiants, nb_etudiants)
    selec_uni = random.sample(universites, nb_uni)

    prefs_etudiants = {
        etu: random.sample(selec_uni, len(selec_uni))
        for etu in selec_etu
    }
    prefs_uni = {
        uni: random.sample(selec_etu, len(selec_etu))
        for uni in selec_uni
    }

    from utils import convert_etu_to_uni_list
    capacites = {uni: capacite for uni in selec_uni}

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


    typer.secho("\n--- Calcul des scores ---", fg=typer.colors.MAGENTA)
    from score import score_final

    
    score_etud_opt = score_final(
        matching_etudiant_opt,
        matching_etudiant_opt,
        matching_universite_opt,
        prefs_etudiants,
        prefs_uni,
        k=k,
        alpha=alpha,
        beta=beta,
        gamma=gamme
    )
    score_univ_opt = score_final(
        matching_etudiant_opt,
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
            matching_etudiant_opt,
            prefs_etudiants,
            prefs_uni,
            capacites
        )

        typer.secho("\n--- Nouveau score après SWING ---", fg=typer.colors.GREEN)

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

        typer.echo("\nScore après SWING :")
        typer.echo(score_swing)

        matching_etudiant_opt_swing = matching_swing

       
    else:
        typer.secho(
            f"\n--- Score étudiant-optimal ≥ 80% ({satisfaction_global_etudopt:.2f}). "
            "SWING non appliqué ---",
            fg=typer.colors.BLUE
        )


    typer.echo("\nScore étudiant-optimal :")
    typer.echo(score_etud_opt)

    typer.echo("\nScore université-optimal :")
    typer.echo(score_univ_opt)


    print(score_etud_opt)




    typer.secho("\n--- Fin ---", fg=typer.colors.YELLOW)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{output_dir}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "preference_et_resultats.json"

    out_data = {
        "preferences": {
            "etudiants": prefs_etudiants,
            "etablissements": prefs_uni,
            "capacites": capacites
        },
        "matchings": {
            "etudiant_optimal": matching_etudiant_opt,
            "universite_optimal": matching_universite_opt
        },
        "scores": {
            "student_optimal": score_etud_opt,
            "university_optimal": score_univ_opt
        }
    }

    if matching_swing is not None:
        out_data["matchings"]["swing"] = matching_swing
        out_data["scores"]["swing"] = score_swing

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    typer.secho(f"\nRésultats sauvegardés dans : {output_path}", fg=typer.colors.GREEN)




if __name__ == "__main__":
    app()