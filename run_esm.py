from pathlib import Path
from typing import List

import esm
import numpy as np
import torch
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.data import Alphabet

# Code based on:
# https://github.com/facebookresearch/esm/blob/main/examples/inverse_folding/sample_sequences.py
# https://github.com/facebookresearch/esm/issues/236

# SET YOUR PARAMETERS HERE
pdbfile = "data/6ZKW.pdb"
outpath = "results/6ZKW.fasta"
chain = "D"
design = ["110D", "111D", "112D", "134D", "135D"]  # 110,111,112,134,135 (chain D)
num_samples = 10
temperature = 1.0
verbose = True


def _seq2index(structure: np.ndarray, design: List[str], target_chain_id: str):
    # Get atoms from design residues
    atoms = [
        f"{atom.res_id}{atom.chain_id}"
        for atom in structure
        if (atom.chain_id == target_chain_id) and (atom.atom_name == "CA")
    ]

    # Get indexes of design residues
    indexes = [index for index, value in enumerate(atoms) if value in design]

    return indexes


def sample_seq_multichain(
    model: GVPTransformerModel,
    alphabet: Alphabet,
    pdbfile: str,
    chain: str,
    design: List[str],
    outpath: str,
    num_samples: int = 1,
    temperature: float = 1.0,
    verbose: bool = False,
):
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
    target_chain_id = chain
    native_seq = native_seqs[target_chain_id]
    print("> Native sequence loaded from structure file:")
    print(native_seq)

    # Sampling sequences with design residues
    samples = []

    # Prepare input for sampling
    target_chain_len = coords[target_chain_id].shape[0]
    all_coords = esm.inverse_folding.multichain_util._concatenate_coords(
        coords, target_chain_id
    )

    # Supply padding tokens for other chains to avoid unused sampling for speed
    padding_pattern = ["<pad>"] * all_coords.shape[0]
    designed_res = _seq2index(structure, design, target_chain_id)

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

    # Sampling
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
        if verbose:
            print(f"Native sequence: {''.join(native_seq[res_id] for res_id in designed_res)}")
            print(f"Designed sequence: {''.join(samples[i][res_id] for res_id in designed_res)}")
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


if __name__ == "__main__":
    # UserWarning: Regression weights not found, predicting contacts will not produce correct results.
    # @tomsercu: You don't need the regression weights, these are for contact prediction only. They are not uploaded on purpose to prevent folks from inadvertently using esm-1v for contact prediction which will lead to poor results, as discussed in the paper.
    # https://github.com/facebookresearch/esm/issues/170#issuecomment-1076687163
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()
    model = model.cuda()

    # Sampling sequences
    sample_seq_multichain(
        model,
        alphabet,
        pdbfile,
        chain,
        design,
        outpath,
        num_samples,
        temperature,
        verbose,
    )
