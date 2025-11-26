from typing import Dict, List


# ------------------------------------------------------------
# REGRET ÉTUDIANTS
# ------------------------------------------------------------
def regret_etudiants(matching: Dict[str, str],
                     matching_ideal_etu: Dict[str, str],
                     prefs_etus: Dict[str, List[str]]) -> float:
    regrets = []

    for etu, prefs in prefs_etus.items():
        n = len(prefs)

        # affectation réelle
        uni = matching.get(etu)
        rank = prefs.index(uni) if uni in prefs else n - 1

        # affectation dans matching optimal étudiant
        ideal_uni = matching_ideal_etu.get(etu)
        ideal_rank = prefs.index(ideal_uni) if ideal_uni in prefs else n - 1

        regret = (rank - ideal_rank) / (n - 1)
        regrets.append(regret)

    return sum(regrets) / len(regrets)


# ------------------------------------------------------------
# REGRET UNIVERSITÉS
# ------------------------------------------------------------
def regret_universites(matching: Dict[str, str],
                       matching_ideal_uni: Dict[str, str],
                       prefs_unis: Dict[str, List[str]]) -> float:
    regrets = []
    # print(matching_ideal_uni)

    # reconstruire : université → étudiant
    uni_to_etu = {}
    uni_to_etu_ideal = {}

    for etu, uni in matching.items():
        uni_to_etu[uni] = etu

    for etu, uni in matching_ideal_uni.items():
        uni_to_etu_ideal[uni] = etu

    for uni, prefs in prefs_unis.items():
        n = len(prefs)

        etu = uni_to_etu.get(uni)
        ideal_etu = uni_to_etu_ideal.get(uni)

        rank = prefs.index(etu) if etu in prefs else n - 1
        ideal_rank = prefs.index(ideal_etu) if ideal_etu in prefs else n - 1

        regret = (rank - ideal_rank) / (n - 1)
        regrets.append(regret)

    return sum(regrets) / len(regrets)


# ------------------------------------------------------------
# REGRET GLOBAL
# ------------------------------------------------------------
def regret_global(matching: Dict[str, str],
                  matching_ideal_etu: Dict[str, str],
                  matching_ideal_uni: Dict[str, str],
                  prefs_etus: Dict[str, List[str]],
                  prefs_unis: Dict[str, List[str]]):



    r_etu = regret_etudiants(matching,
                             matching_ideal_etu,
                             prefs_etus)

    r_uni = regret_universites(matching,
                               matching_ideal_uni,
                               prefs_unis)

    # print((r_etu, r_uni, (r_etu + r_uni) / 2))
    return (r_etu, r_uni, (r_etu + r_uni) / 2)
