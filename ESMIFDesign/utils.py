import json
from typing import Dict, List


def get_chains(design: List[str]) -> List[str]:
    # Get all chains
    chains = [residue[-1] for residue in design]

    # Get ordered unique chains
    chains = list(dict.fromkeys(chains))

    return chains


def get_frequency_of_residues(designs: List[str], num_samples: int) -> Dict[str, int]:
    AA = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]

    frequency = {}
    for aa in AA:
        frequency[aa] = [item.count(aa) / num_samples for item in zip(*designs)]

    return frequency


def read_config(filepath: str) -> Dict[str, List[str]]:
    with open(filepath, "r") as f:
        config = json.load(f)
    return config
