"""
Script to generate resources/rules_v2.yaml from classification/hybrid_reasoning.py _RULE_SPECS.
"""

from __future__ import annotations

import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.hybrid_reasoning import _RULE_SPECS

YAML_PATH = ROOT / "resources" / "rules_v2.yaml"

def generate_yaml():
    rules_by_pramana = {
        "Pratyaksha": [],
        "Shabda": [],
        "Upamana": [],
        "Anumana": [],
    }

    counts = {"Pratyaksha": 0, "Shabda": 0, "Upamana": 0, "Anumana": 0}

    for pattern, weight, pramana, cue in _RULE_SPECS:
        counts[pramana] += 1
        rule_id = f"{pramana.upper()}_{counts[pramana]:03d}"
        
        rule_entry = {
            "id": rule_id,
            "pramana": pramana,
            "cue_type": cue,
            "weight": weight,
            "pattern": pattern,
            "description": f"Symbolic cue pattern for {pramana} ({cue}).",
            "example_trigger": f"Example sentence containing {cue}.",
            "example_non_trigger": f"Example sentence without {cue}.",
        }
        rules_by_pramana[pramana].append(rule_entry)

    data = {
        "version": "2.0",
        "total_rules": len(_RULE_SPECS),
        "generated_from": "classification/hybrid_reasoning.py::_RULE_SPECS",
        "date": "2026-08-08",
        "Pratyaksha": rules_by_pramana["Pratyaksha"],
        "Shabda": rules_by_pramana["Shabda"],
        "Upamana": rules_by_pramana["Upamana"],
        "Anumana": rules_by_pramana["Anumana"],
    }

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print(f"Successfully generated {YAML_PATH} with {len(_RULE_SPECS)} rules!")

if __name__ == "__main__":
    generate_yaml()
