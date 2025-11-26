import json
import random
from pathlib import Path
from typing import Dict, List
import typer
from utils import load_data_from_json
from datetime import datetime

app = typer.Typer()

def gale_shapley_etudiant_optimal(prefs_etus: Dict[str, List[str]],
                                 prefs_unis: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Implémentation standard 1-to-1 du Gale-Shapley étudiant-proposant.
    Retourne: dict etudiant -> université
    """

    free_students = list(prefs_etus.keys())
    next_proposal = {s: 0 for s in prefs_etus}

    # matching inversé : uni -> étudiant (ou None)
    uni_partner = {u: None for u in prefs_unis}

    # matching final étudiant -> université
    etu_partner = {s: None for s in prefs_etus}

    while free_students:
        s = free_students.pop(0)
        prefs = prefs_etus[s]

        # Si étudiant a épuisé sa liste, il reste sans match
        if next_proposal[s] >= len(prefs):
            continue

        u = prefs[next_proposal[s]]
        next_proposal[s] += 1

        current = uni_partner[u]

        # 1) Université libre → accepte
        if current is None:
            uni_partner[u] = s
            etu_partner[s] = u

        else:
            # 2) Université compare les deux étudiants
            ranking = prefs_unis[u]

            if ranking.index(s) < ranking.index(current):
                # université préfère le nouvel étudiant s
                uni_partner[u] = s
                etu_partner[s] = u

                # l'ancien devient libre
                etu_partner[current] = None
                free_students.append(current)

            else:
                # université garde son étudiant → s reste libre
                free_students.append(s)

    return etu_partner


# def gale_shapley_etudiant_optimal(prefs_etus: Dict[str, List[str]],
#                                  prefs_unis: Dict[str, List[str]],
#                                  capacities: Dict[str, int]) -> Dict[str, List[str]]:
    
#     free = list(prefs_etus.keys())

#     next_dem =  {s: 0 for s in prefs_etus}
#     matching = {u: [] for u in prefs_unis}

#     while free:
#         etu = free.pop(0)
#         prefs = prefs_etus[etu]

#         if next_dem[etu] >= len(prefs): continue

#         uni = prefs[next_dem[etu]]
#         next_dem[etu] += 1

#         if len(matching[uni]) < capacities[uni]:
#             matching[uni].append(etu)
#         else:
#             classement = prefs_unis[uni]
#             pire = max(matching[uni], key = lambda x: classement.index(x))

#             if classement.index(etu) < classement.index(pire):
#                 matching[uni].remove(pire)
#                 matching[uni].append(etu)
#                 free.append(pire)
#             else: free.append(etu)

#     return matching

def gale_shapley_university_optimal(prefs_etus: Dict[str, List[str]],
                                    prefs_unis: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Implémentation standard 1-to-1 du Gale-Shapley université-proposante.
    Retourne: dict etudiant -> université
    """

    free_unis = list(prefs_unis.keys())
    next_proposal = {u: 0 for u in prefs_unis}

    # partenaire de chaque étudiant
    etu_partner = {s: None for s in prefs_etus}
    # partenaire de chaque université
    uni_partner = {u: None for u in prefs_unis}

    while free_unis:
        u = free_unis.pop(0)
        prefs = prefs_unis[u]

        if next_proposal[u] >= len(prefs):
            continue

        s = prefs[next_proposal[u]]
        next_proposal[u] += 1

        current = etu_partner[s]

        # 1) Étudiant libre → accepte
        if current is None:
            etu_partner[s] = u
            uni_partner[u] = s

        else:
            # 2) Étudiant compare les deux universités
            ranking = prefs_etus[s]

            if ranking.index(u) < ranking.index(current):
                # étudiant préfère la nouvelle université
                etu_partner[s] = u
                uni_partner[u] = s

                # ancienne université devient libre
                uni_partner[current] = None
                free_unis.append(current)

            else:
                # étudiant préfère son université actuelle → u reste libre
                free_unis.append(u)

    return etu_partner

# def gale_shapley_university_optimal(prefs_etus: Dict[str, List[str]],
#                                     prefs_unis: Dict[str, List[str]],
#                                     capacities: Dict[str, int]) -> Dict[str, List[str]]:
    
#     free = list(prefs_unis.keys())
#     next_dem = {u: 0 for u in prefs_unis}
#     matching = {u: [] for u in prefs_unis}
#     etus_assign = {s: None for s in prefs_etus}

#     while free:
#         uni = free.pop(0)
#         if next_dem[uni] >= len(prefs_unis[uni]):
#             continue

#         etu = prefs_unis[uni][next_dem[uni]]
#         next_dem[uni] += 1

#         current = etus_assign[etu]

#         if current is None:
#             etus_assign[etu] = uni
#             matching[uni].append(etu)
#         else:
#             classement = prefs_etus[etu]
#             if classement.index(uni) < classement.index(current):
#                 matching[current].remove(etu)
#                 matching[uni].append(etu)
#                 etus_assign[etu] = uni
#                 free.append(current)
#             else:
#                 free.append(uni)

#     return matching

# def est_stable(matching: Dict[str, List[str]],
#                prefs_etus: Dict[str, List[str]],
#                prefs_unis: Dict[str, List[str]]):

#     # affectation étu -> uni
#     assigned = {}
#     for uni, etus in matching.items():
#         for s in etus:
#             assigned[s] = uni

#     for etu, prefs in prefs_etus.items():
#         assigned_uni = assigned.get(etu)

#         for uni in prefs:
#             if uni == assigned_uni:
#                 break

#             assigned_list = matching[uni]
#             classement = prefs_unis[uni]

#             if not assigned_list:
#                 return False, (etu, uni)

#             pire = max(assigned_list, key=lambda x: classement.index(x))

#             if classement.index(etu) < classement.index(pire):
#                 return False, (etu, uni)

#     return True, None

def est_stable(
    matching,
    prefs_etus,
    prefs_unis
):
    """
    Vérifie la stabilité d'un matching 1-to-1.
    Le matching peut être sous plusieurs formes :
        - étudiant -> université
        - université -> étudiant
        - université -> [étudiant]
    Cette fonction détecte automatiquement le format et convertit correctement.
    """

    # -----------------------------
    # 1) Détection automatique du format
    # -----------------------------
    etu_to_uni = {}

    first_key = next(iter(matching.keys()))

    # Cas A : matching[etu] = uni  (clé = étudiant)
    if first_key in prefs_etus:
        for etu, uni in matching.items():
            if isinstance(uni, list):  # au cas ou tu as ['u']
                if len(uni) == 1:
                    etu_to_uni[etu] = uni[0]
                else:
                    raise ValueError("Matching non 1-to-1.")
            else:
                etu_to_uni[etu] = uni

    # Cas B : matching[uni] = etu  (clé = université)
    elif first_key in prefs_unis:
        for uni, etu in matching.items():
            if isinstance(etu, list):
                if len(etu) == 1:
                    etu = etu[0]
                else:
                    raise ValueError("Matching non 1-to-1.")
            etu_to_uni[etu] = uni

    else:
        raise ValueError("Matching n'est ni etudiant->universite ni universite->etudiant.")

    # -----------------------------
    # 2) Vérification stabilité
    # -----------------------------
    for etu, prefs in prefs_etus.items():

        assigned_uni = etu_to_uni.get(etu)

        for uni in prefs:
            if uni == assigned_uni:  
                break  # on arrête car le reste est moins préféré

            # université libre ?
            current_student = None
            # reconstruire mapping inverse uni->etu
            for e, u in etu_to_uni.items():
                if u == uni:
                    current_student = e
                    break

            if current_student is None:
                return False, (etu, uni)

            ranking = prefs_unis[uni]

            # Si université préfère cet étudiant
            if ranking.index(etu) < ranking.index(current_student):
                return False, (etu, uni)

    return True, None
