import json 
import typer
from pathlib import Path

def load_data_from_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

        