from pathlib import Path
from typing import Dict, List


def top_k_etus(matching_etu_to_uni: Dict[str, str],
               prefs_etus: Dict[str, List[str]],
               k: int):
    """
    matching_etu_to_uni : dict étudiant -> université
    prefs_etus         : dict étudiant -> [universités]
    """
    scores = []

    for etu, prefs in prefs_etus.items():

        if etu not in matching_etu_to_uni:
            scores.append(0)
            continue

        uni = matching_etu_to_uni[etu]

        # rang de l'université pour l'étudiant
        rang = prefs.index(uni)

        if rang < k:
            score = 1 - (rang / k)     # satisfaction décroissante
        else:
            score = 0

        scores.append(score)

    return sum(scores) / len(prefs_etus)

def top_k_unis(matching_etu_to_uni: Dict[str, str],
               prefs_unis: Dict[str, List[str]],
               k: int):
    """
    matching_etu_to_uni : dict étudiant -> université
    prefs_unis          : dict université -> [étudiants]
    """
    scores = []

    for etu, uni in matching_etu_to_uni.items():

        
        prefs_uni = prefs_unis[uni]

        rang = prefs_uni.index(etu)

        if rang < k:
            score = 1 - (rang / k)
        else:
            score = 0

        scores.append(score)

    return sum(scores) / len(scores) if scores else 0


def top_k_global(matching_etu_to_uni: Dict[str, str],
                 prefs_unis: Dict[str, List[str]],
                 prefs_etus: Dict[str, List[str]],
                 k: int = 1, alpha: float = 0.9):
    """
    Retourne (score_total, score_etudiants, score_universites)
    """
    score_etu = top_k_etus(matching_etu_to_uni, prefs_etus, k)
    score_uni = top_k_unis(matching_etu_to_uni, prefs_unis, k)

    score = alpha * score_etu + (1 - alpha) * score_uni
    # print("score top k", score)
    return score, score_etu, score_uni


