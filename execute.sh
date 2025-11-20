#!/bin/bash

# === CONFIG ===
PY_SCRIPT="src/generate.py"          # nom du script Python contenant ta commande Typer
ETU_JSON="data/etudiants.json"    # fichier JSON des étudiants
UNI_JSON="data/etablissements.json"  # fichier JSON des universités
OUT_DIR="experiments_final"  # répertoire de sortie global

# Vérifie que les fichiers existent
if [ ! -f "$PY_SCRIPT" ]; then
    echo "❌ ERREUR : le fichier $PY_SCRIPT est introuvable."
    exit 1
fi

if [ ! -f "$ETU_JSON" ]; then
    echo "❌ ERREUR : le fichier $ETU_JSON est introuvable."
    exit 1
fi

if [ ! -f "$UNI_JSON" ]; then
    echo "❌ ERREUR : le fichier $UNI_JSON est introuvable."
    exit 1
fi

echo "🚀 Début de l'exécution des expériences (1 à 33 étudiants et établissements)..."

# === BOUCLES 1 → 33 ===
for etu in $(seq 1 33); do
    
        echo "➡️  Étudiants = $etu, Établissements = $etu"

        # Exécute en passant les paramètres automatiquement
        python3 "$PY_SCRIPT" \
            "$ETU_JSON" \
            "$UNI_JSON" \
            --output-dir "$OUT_DIR" <<EOF
$etu
$etu
1
EOF

done

echo "🎉 Toutes les expériences sont terminées !"
