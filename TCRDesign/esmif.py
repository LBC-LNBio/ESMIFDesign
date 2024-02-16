from pathlib import Path
from typing import Dict, List, Tuple

import esm
import numpy as np
import torch
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.data import Alphabet

# Code based on:
# https://github.com/facebookresearch/esm/blob/main/examples/inverse_folding/sample_sequences.py
# https://github.com/facebookresearch/esm/issues/236


def _concatenate_multichain_coords(
    coords: Dict[str, np.ndarray], target_chain_ids: List[str], padding_length: int = 10
) -> Tuple[np.ndarray, List[str]]:
    # Padding coordinates between concatenated chains
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)

    # For best performance, put the target chains first in concatenation.
    coords_list = []
    for chain_id in target_chain_ids:
        if len(coords_list) > 0:
            coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])

    # Concatenate remaining chains
    for chain_id in coords:
        if chain_id in target_chain_ids:
            continue
        coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])

    # Concatenate all chains
    coords_concatenated = np.concatenate(coords_list, axis=0)

    return coords_concatenated


def _seq2index(
    structure: np.ndarray, design: List[str], target_chain_ids: List[str]
) -> List[int]:
    # Get atoms from design residues
    atoms = [
        f"{atom.res_id}{atom.chain_id}"
        for atom in structure
        if (atom.chain_id in target_chain_ids) and (atom.atom_name == "CA")
    ]

    # Get indexes of design residues
    indexes = [index for index, value in enumerate(atoms) if value in design]

    return indexes


def sample_seq_multichain(
    model: GVPTransformerModel,
    alphabet: Alphabet,
    pdbfile: str,
    chains: str,
    design: List[str],
    outpath: str,
    num_samples: int = 1,
    temperature: float = 1.0,
    verbose: bool = False,
) -> Tuple[List[str], List[float]]:
    # Transfer model to GPU if available
    if torch.cuda.is_available():
        print("> Transferring model to GPU ...")
        model = model.cuda()

    # Load structure
    structure = esm.inverse_folding.util.load_structure(pdbfile)
    coords, native_seqs = (
        esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    )

    # Load native sequences
    target_chain_ids = chains
    native_seq = "".join([native_seqs[chain_id] for chain_id in target_chain_ids])
    print("> Native sequence loaded from structure file:")
    print(native_seq)

    # Prepare input for sampling
    target_chain_len = 0
    for chain_id in target_chain_ids:
        target_chain_len += coords[chain_id].shape[0]
    all_coords = _concatenate_multichain_coords(
        coords, target_chain_ids, padding_length=10
    )

    # Supply padding tokens for other chains to avoid unused sampling for speed
    padding_pattern = ["<pad>"] * all_coords.shape[0]
    designed_res = _seq2index(structure, design, target_chain_ids)

    # <res_name> for design residues
    # <mask> for designed residues
    # <pad> to ignore other chains
    for n in range(target_chain_len):
        if n not in designed_res:
            padding_pattern[n] = native_seq[n]
        else:
            padding_pattern[n] = "<mask>"

    # Send coordinates to gpu
    all_coords = torch.from_numpy(all_coords).to("cuda:0")

    # Sampling sequences with design residues
    samples, recoveries = [], []

    for i in range(num_samples):
        print(f"\n> Sampling.. ({i+1} of {num_samples})")
        sampled = model.sample(
            all_coords,
            partial_seq=padding_pattern,
            temperature=temperature,
            device="cuda:0",
        )
        sampled_seq = sampled[:target_chain_len]
        # end design residues

        # Append sampled sequence to list
        samples.append(sampled_seq)
        if verbose:
            print(f"Sampled sequence {i+1}:")
            print(samples[i])

        # Sequence recovery
        recovery = np.mean(
            [
                (a == b)
                for a, b in zip(
                    "".join(native_seq[res_id] for res_id in designed_res),
                    "".join(samples[i][res_id] for res_id in designed_res),
                )
            ]
        )
        recoveries.append(recovery)
        if verbose:
            print(
                f"Native sequence: {''.join(native_seq[res_id] for res_id in designed_res)}"
            )
            print(
                f"Designed sequence: {''.join(samples[i][res_id] for res_id in designed_res)}"
            )
        print("Sequence recovery:", recovery)

    # Save sampled sequences to file
    print(f"\n> Saving sampled sequences to {outpath}.")
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write(">native_seq\n")
        f.write(native_seq + "\n")
        for i in range(num_samples):
            f.write(f">sampled_seq_{i+1}\n")
            f.write(samples[i] + "\n")

    return samples, recoveries