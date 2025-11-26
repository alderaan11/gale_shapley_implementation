from typing import Dict, List
import math

def satisfaction_individuelle(rang: int, taille: int) -> float:
    if taille <= 1:
        return 1.0
    return (taille - rang - 1) / (taille - 1)


def satisfaction_croisee_globale(
    matching_etu_to_uni: Dict[str, str],
    prefs_etus: Dict[str, List[str]],
    prefs_unis: Dict[str, List[str]]
) -> float:
    """
    matching_etu_to_uni : dict étudiant -> université (FORMAT DE TON GALE-SHAPLEY)
    prefs_etus : étudiant -> liste d'universités
    prefs_unis : université -> liste d'étudiants
    """

    scores = []

    # Parcours des couples (étudiant, université)
    for etu, uni in matching_etu_to_uni.items():

        # ---------- Satisfaction de l'étudiant ----------
        prefs_etu = prefs_etus[etu]
        taille_etu = len(prefs_etu)

        if uni not in prefs_etu:
            S_etu = 0.0
        else:
            rang_etu = prefs_etu.index(uni)
            S_etu = satisfaction_individuelle(rang_etu, taille_etu)

        # ---------- Satisfaction de l'université ----------
        prefs_uni = prefs_unis[uni]
        taille_uni = len(prefs_uni)

        if etu not in prefs_uni:
            S_uni = 0.0
        else:
            rang_uni = prefs_uni.index(etu)
            S_uni = satisfaction_individuelle(rang_uni, taille_uni)

        # ---------- Satisfaction croisée ----------
        if S_etu <= 0 or S_uni <= 0:
            S_cross = 0.0
        else:
            S_cross = (S_etu + S_uni) / 2  # correcte

        scores.append(S_cross)

    if not scores:
        return 0.0
    # print(sum(scores) / len(scores))
    return sum(scores) / len(scores)
