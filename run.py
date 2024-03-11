import os
import warnings

import pandas as pd
import torch

from TCRDesign import (
    esm,
    get_chains,
    get_frequency_of_residues,
    prepare_sample_output,
    read_config,
    sample_seq_multichain,
)

# Set seed
torch.manual_seed(37)

# Just suppress all warnings with this:
warnings.filterwarnings("ignore")

# CONSTANTS
NUM_SAMPLES = 10
TEMPERATURE = 0.2
PADDING = 10
VERBOSE = False

if __name__ == "__main__":
    # UserWarning: Regression weights not found, predicting contacts will not produce correct results.
    # @tomsercu: You don't need the regression weights, these are for contact prediction only. They are not uploaded on purpose to prevent folks from inadvertently using esm-1v for contact prediction which will lead to poor results, as discussed in the paper.
    # https://github.com/facebookresearch/esm/issues/170#issuecomment-1076687163
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()

    # Read configuration file
    config = read_config("config.json")

    # Create summary
    summary = {}
    summary["design"] = {}
    summary["recovery"] = {}
    summary["uniqueness"] = {}
    summary["frequency"] = {}

    # Iterate through all PDB files
    for pdb in config:
        print(f"[==> {pdb}")

        # Prepare parameters
        basedir = os.path.join("results")
        pdbfile = os.path.join("data", f"{pdb}.pdb")
        outpath = os.path.join(basedir, f"{pdb}.fasta")
        design = config[pdb]
        chains = get_chains(design)

        # Sampling sequences
        samples, recoveries = sample_seq_multichain(
            model,
            alphabet,
            pdbfile,
            chains,
            design,
            outpath,
            NUM_SAMPLES,
            TEMPERATURE,
            PADDING,
            VERBOSE,
        )

        # Save samples
        summary["design"][pdb] = prepare_sample_output(
            samples, pdbfile, chains, design, PADDING, basedir
        )

        # Save recovery
        summary["recovery"][pdb] = recoveries

        # Save uniqueness
        # Uniqueness = number of unique designs / total number of designs
        summary["uniqueness"][pdb] = [
            len(list(set(summary["design"][pdb]))) / NUM_SAMPLES
        ]

        # Save frequency per position
        summary["frequency"][pdb] = get_frequency_of_residues(
            summary["design"][pdb], NUM_SAMPLES
        )

    # Convert designs to pandas DataFrame
    samples = pd.DataFrame(summary["design"])
    samples.to_csv(os.path.join(basedir, "designs.csv"))

    # Convert recoveries to pandas DataFrame
    recoveries = pd.DataFrame(summary["recovery"])
    recoveries.to_csv(os.path.join(basedir, "recoveries.csv"))

    # Convert uniqueness to pandas DataFrame
    uniqueness = pd.DataFrame(summary["uniqueness"])
    uniqueness.to_csv(os.path.join(basedir, "uniqueness.csv"))

    # Convert frequency to pandas DataFrame
    frequency = pd.DataFrame(summary["frequency"])
    frequency.to_csv(os.path.join(basedir, "frequency.csv"))
    # Show summary to user
    print(summary)
