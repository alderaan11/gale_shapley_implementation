from pathlib import Path
import random
import json
from loader import (
    load_students_json,
    load_universities_json,
    extract_preferences_students,
    extract_preferences_universities
)
from stable_marriage import (
    gale_shapley_student_optimal,
    gale_shapley_university_optimal,
    is_stable,
    satisfaction,
    normalize_name
)


def main():
    print("=== Chargement des données JSON ===")

    path_students = Path("data/etudiants.json")
    path_unis = Path("data/etablissements.json")

    students_map = load_students_json(path_students)
    unis_map = load_universities_json(path_unis)

    all_students = list(students_map.keys())
    all_unis = list(unis_map.keys())

    n = int(input("Nombre d'étudiants à sélectionner : "))
    m = int(input("Nombre d'établissements à sélectionner : "))
    cap = int(input("Capacité des établissements : "))

    selected_students = random.sample(all_students, n)
    selected_unis = random.sample(all_unis, m)

    # Charger les préférences complètes depuis un JSON externe
    prefs_full = json.loads(Path("/home/e20200011242/Bureau/gale_shapley_implementation/experiments/2025-11-19 15:45:49.669637/preference.json").read_text())

    prefs_students = extract_preferences_students(
        prefs_full["etudiants"],
        selected_students,
        selected_unis
    )

    prefs_unis = extract_preferences_universities(
        prefs_full["etablissements"],
        selected_students,
        selected_unis
    )

    capacities = {u: cap for u in selected_unis}

    print("\n=== Gale–Shapley étudiant-optimal ===")
    match1 = gale_shapley_student_optimal(prefs_students, prefs_unis, capacities)
    stable1, pair1 = is_stable(match1, prefs_students, prefs_unis)
    sat_s1, sat_u1 = satisfaction(match1, prefs_students, prefs_unis)

    print("Stable :", stable1)
    print("Satisfaction étudiants :", sat_s1)
    print("Satisfaction universités :", sat_u1)

    print("\n=== Gale–Shapley université-optimal ===")
    match2 = gale_shapley_university_optimal(prefs_students, prefs_unis, capacities)
    stable2, pair2 = is_stable(match2, prefs_students, prefs_unis)
    sat_s2, sat_u2 = satisfaction(match2, prefs_students, prefs_unis)

    print("Stable :", stable2)
    print("Satisfaction étudiants :", sat_s2)
    print("Satisfaction universités :", sat_u2)


if __name__ == "__main__":
    main()
