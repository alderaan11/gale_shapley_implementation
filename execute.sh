#!/bin/bash

for i in {1..10}
do
    echo "---- Exécution $i ----"
    python3 src/main.py data/etudiants.json data/etablissements.json
    echo ""
done
