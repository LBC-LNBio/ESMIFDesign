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
) -> np.ndarray:
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
    padding_length: int = 10,
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

    # Prepare input for sampling
    all_coords = _concatenate_multichain_coords(
        coords, chains, padding_length=padding_length
    )

    # Get all_coords chain ordering
    all_coords_chains = chains + list(set(native_seqs.keys()) - set(chains))

    # Get chain sizes
    chain_sizes = [len(native_seqs[chain_id]) for chain_id in all_coords_chains]

    # Get length of target chain
    target_chain_len = 0
    for chain_id, size in zip(all_coords_chains, chain_sizes):
        if chain_id in chains:
            target_chain_len += size + padding_length

    # Get native sequence
    native_seq = []
    for i, chain_id in enumerate(all_coords_chains):
        for j in range(chain_sizes[i]):
            native_seq.append(native_seqs[chain_id][j])
        if i < len(all_coords_chains) - 1:
            for j in range(padding_length):
                native_seq.append("-")

    # Prepare design indexes
    indexes = []
    for i, chain_id in enumerate(all_coords_chains):
        start = sum([chain_sizes[j] for j in range(i)])
        index = [
            index + start + (i * padding_length)
            for index in _seq2index(structure, design, chain_id)
        ]
        if len(index) > 0:
            indexes.extend(index)

    # Supply padding tokens for other chains to avoid unused sampling for speed
    # <res_name> for fixed residues
    # <mask> for designed residues
    # <pad> to ignore other chains
    padding_pattern = []
    for i, chain_id in enumerate(all_coords_chains):
        for j in range(chain_sizes[i]):
            padding_pattern.append(native_seqs[chain_id][j])
        if i < len(all_coords_chains) - 1:
            for j in range(padding_length):
                padding_pattern.append("<pad>")
    for index in indexes:
        padding_pattern[index] = "<mask>"

    # Send coordinates to gpu
    if torch.cuda.is_available():
        print("> Transferring data to GPU ...")
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
        sampled = sampled.replace("<pad>", "-")

        if verbose:
            print(f"> Sampled sequence {i+1}:")
            print(sampled)
            print("".join(padding_pattern).replace("<mask>", "X").replace("<pad>", "-"))

        # Append samples sequence to list
        samples.append(sampled[:target_chain_len])

        # Sequence recovery
        recovery = np.mean(
            [
                (a == b)
                for a, b in zip(
                    "".join(native_seq[res_id] for res_id in indexes),
                    "".join(samples[i][res_id] for res_id in indexes),
                )
            ]
        )
        recoveries.append(recovery)
        print(
            f"Native sequence: {''.join(native_seq[res_id] for res_id in indexes)}"
        )
        print(
            f"Designed sequence: {''.join(samples[i][res_id] for res_id in indexes)}"
        )
        print("Sequence recovery:", recovery)

    # Save sampled sequences to file
    print(f"\n> Saving sampled sequences to {outpath}.")
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write(">native_seq\n")
        f.write("".join(native_seq) + "\n")
        for i in range(num_samples):
            f.write(f">sampled_seq_{i+1}\n")
            f.write(samples[i] + "\n")

    return samples, recoveries
