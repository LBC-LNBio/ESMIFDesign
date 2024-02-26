import json
from typing import Dict, List


def get_chains(design: List[str]) -> List[str]:
    # Get all chains
    chains = [residue[-1] for residue in design]

    # Get ordered unique chains
    chains = list(dict.fromkeys(chains))

    return chains


def read_config(filepath: str) -> Dict[str, List[str]]:
    with open(filepath, "r") as f:
        config = json.load(f)
    return config
