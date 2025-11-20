import json
from pathlib import Path
import typer

from stable_marriage import (
    gale_shapley,
    calculer_satisfaction,
    est_stable
)

app = typer.Typer(help="Exécute les deux versions de Gale–Shapley sur un fichier de préférences.")


def load_json(path: Path):
    if not path.exists():
        typer.secho(f"Erreur : fichier introuvable → {path}", fg="red")
        raise typer.Exit(1)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.command()
def run(preferences_json: Path = typer.Argument(...)):
    """
    1) upload des préférences depuis JSON
    2) GS étudiant-optimal
    3) GS université-optimal (correctement inversé)
    4) test de stabilité
    5) satisfaction
    """

    data = load_json(preferences_json)

    prefs_etudiants = data["etudiants"]
    prefs_universites = data["etablissements"]
    capacites_uni = data["capacites"]

    # ----------------------------------------------------------------------
    # 2) GALE–SHAPLEY côté ÉTUDIANTS
    # ----------------------------------------------------------------------
    print("\n===== GALE–SHAPLEY : étudiant-optimal =====")

    matching_etu = gale_shapley(
        prefs_etudiants,
        prefs_universites,
        capacites_uni
    )

    for uni, etus in matching_etu.items():
        print(f"{uni} ← {etus}")

    stable1, bloc1 = est_stable(matching_etu, prefs_etudiants, prefs_universites)
    print(f"Stabilité : {stable1}")
    if not stable1:
        print(f"Paire bloquante : {bloc1}")

    sati_etu_1, sati_uni_1 = calculer_satisfaction(matching_etu, prefs_etudiants, prefs_universites)
    print(f"Satisfaction étudiants : {sati_etu_1:.3f}")
    print(f"Satisfaction universités : {sati_uni_1:.3f}")

    # ----------------------------------------------------------------------
    # 3) GALE–SHAPLEY côté UNIVERSITÉS (version correcte)
    # ----------------------------------------------------------------------
    print("\n===== GALE–SHAPLEY : université-optimal =====")

    # Universités = "proposants"
    prefs_uni_proposants = prefs_universites
    prefs_etu_receveurs = prefs_etudiants

    # Capacités : universités gardent leur capacité, étudiants = 1
    capacites_uni_prop = capacites_uni.copy()
    capacites_etudiants = {etu: 1 for etu in prefs_etudiants}

    # On utilise GS mais en disant :
    # - "étudiants" = universités
    # - "établissements" = étudiants
    matching_raw = gale_shapley(
        prefs_etudiants=prefs_uni_proposants,
        prefs_etablissements=prefs_etu_receveurs,
        capacites=capacites_uni_prop
    )

    # Reconstruction vers format universités → liste d’étudiants
    matching_uni = {uni: [] for uni in prefs_universites}
    for etu, uni in matching_raw.items():
        matching_uni[uni].append(etu)

    for uni, etus in matching_uni.items():
        print(f"{uni} ← {etus}")

    stable2, bloc2 = est_stable(matching_uni, prefs_etudiants, prefs_universites)
    print(f"Stabilité : {stable2}")
    if not stable2:
        print(f"Paire bloquante : {bloc2}")

    sati_etu_2, sati_uni_2 = calculer_satisfaction(matching_uni, prefs_etudiants, prefs_universites)
    print(f"Satisfaction étudiants : {sati_etu_2:.3f}")
    print(f"Satisfaction universités : {sati_uni_2:.3f}")

    print("\n===== FIN =====")


if __name__ == "__main__":
    app()
