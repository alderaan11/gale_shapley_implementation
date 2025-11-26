from typing import List, Dict
from pathlib import Path
from satisfaction import satisfaction_croisee_globale
from top_k import top_k_etus, top_k_unis, top_k_global
from frustration import frustration_etudiants, frustration_etablissements
from regret import regret_global
def score_final(
    matching: Dict[str, List[str]],
    matching_etu: Dict[str, List[str]],
    matching_uni: Dict[str, List[str]],
    prefs_etus: Dict[str, List[str]],
    prefs_unis: Dict[str, List[str]], 
    k: int,
    alpha: float,
    beta: float,
    gamma: float,
):
# 
    # k=1
    # alpha=0.15
    # beta=0.15
    # gamma=0.70

    topk_etu = top_k_etus(matching, prefs_etus, k)
    topk_uni = top_k_unis(matching, prefs_unis, k)
    # topk_global = (topk_etu + topk_uni) / 2
    topk_etus, topk_unis, topk_global = top_k_global(matching, prefs_unis, prefs_etus)
    S_cross = satisfaction_croisee_globale(matching, prefs_etus, prefs_unis)


    r_etu, r_uni, r_global = regret_global(matching, matching_etu, matching_uni, prefs_etus, prefs_unis)
    print("top_global:", topk_global, topk_etus, topk_unis)
    print("satisfaction:", S_cross)
    print("regret global: ", r_global)
    

    optimality = 1 - r_global
    # score = (
    #     alpha * topk_global +
    #     beta * S_cross +
    #     gamma * stab_global
    # )
    score = (
        alpha * topk_global +
        beta * S_cross +
        gamma * optimality
    )
    return {
        "score_final": score,
        "topk_global": topk_global,
        "satisfaction_croisee": S_cross,
        "regret_global":r_global,
        "optimality": optimality,
        "details": {
            "k": k,
            "topk_etu": topk_etu,
            "topk_uni": topk_uni,
            "r_etu": r_etu,
            "r_uni": r_uni,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma
        }
    }
