import json
import random
from stable_marriage import normalize_name


def load_students_json(path):
    data = json.loads(path.read_text())
    raw = data["nom"]
    return {normalize_name(x): x for x in raw}


def load_universities_json(path):
    data = json.loads(path.read_text())
    raw = [u["nom"] for u in data["etablissements_superieurs_francais_complet"]]
    return {normalize_name(x): x for x in raw}


def extract_preferences_students(pref_json, selected_unis):
    result = {}
    for stu, raw_prefs in pref_json.items():
        clean = [u for u in raw_prefs if u in selected_unis]
        result[stu] = clean
    return result

def extract_preferences_universities(pref_json, selected_students, selected_unis):
    result = {}
    for uni in selected_unis:
        raw_prefs = pref_json[uni]
        clean = [s for s in raw_prefs if s in selected_students]
        result[uni] = clean
    return result
