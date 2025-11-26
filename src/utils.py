import json
from pathlib import Path


def load_data_from_json(json_path:Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def convert_etu_to_uni_list(matching_etu_to_uni, prefs_unis):
    """
    Convertit un matching etudiant -> universite
    en matching universite -> [etudiant]
    (1-to-1 uniquement)
    """
    matching_uni_to_list = {u: [] for u in prefs_unis.keys()}

    for etu, uni in matching_etu_to_uni.items():
        matching_uni_to_list[uni] = [etu]

    return matching_uni_to_list
