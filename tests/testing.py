import os
import sys
import warnings
from typing import List

import pandas as pd
import torch
from esm.data import Alphabet
from esm.inverse_folding.gvp_transformer import GVPTransformerModel

sys.path.append("../")

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


def testing_6ZKW(model: GVPTransformerModel, alphabet: Alphabet):
    print("====== 6ZKW ======\n")
    # Define parameters
    pdbs = [
        os.path.join("data", "6ZKW.pdb"),
        os.path.join("data", "6ZKW_DE.pdb"),
        os.path.join("data", "6ZKW_contact_to_gly.pdb"),
        os.path.join("data", "6ZKW_all_gly.pdb"),
    ]
    chains = ["D", "E"]
    # 110,111,112,134,135 (chain D)
    # 113,114,133 (chain E)
    design = ["110D", "111D", "112D", "134D", "135D", "113E", "114E", "133E"]

    # Summary of recoveries
    summary = {}

    # Sampling sequences in pdbs
    for pdbfile in pdbs:
        print(f"> {pdbfile}")
        outpath = os.path.join(
            "results", f"{os.path.basename(pdbfile).replace('.pdb', '.fasta')}"
        )

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

        # Append recoveries to summary
        recoveries.append(sum(recoveries) / len(recoveries))
        summary[pdbfile] = recoveries

    # Print summary
    summary = pd.DataFrame(summary)
    summary.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Average"]
    summary.to_csv(os.path.join("results", "6ZKW", "summary.csv"))
    print(summary)


def testing_pMHC(model: GVPTransformerModel, alphabet: Alphabet):
    print("====== pMHC ======\n")
    # Read configuration file
    config = read_config("pMHC1.json")

    # Create directory
    basedir = os.path.join("results", "pMHC")
    os.makedirs(basedir, exist_ok=True)
    os.makedirs(os.path.join(basedir, "pMHC"), exist_ok=True)
    os.makedirs(os.path.join(basedir, "no_pMHC"), exist_ok=True)

    # Create summary
    summary = {}
    summary["pMHC"] = {"design": {}, "recovery": {}, "uniqueness": {}, "frequency": {}}
    summary["no_pMHC"] = {
        "design": {},
        "recovery": {},
        "uniqueness": {},
        "frequency": {},
    }

    for pdb in config:
        print(f"[==> {pdb}")

        for datatype in ["pMHC", "no_pMHC"]:
            print(f"> {datatype}")

            # Prepare parameters
            pdbfile = os.path.join("data", "pMHC", datatype, f"{pdb}.pdb")
            outpath = os.path.join(basedir, datatype, f"{pdb}.fasta")
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

            # Save recovery
            summary[datatype][pdb] = recoveries

            # Save samples
            summary[datatype]["design"][pdb] = prepare_sample_output(
                samples,
                pdbfile,
                chains,
                design,
                PADDING,
                os.path.join(basedir, datatype),
            )

            # Save recovery
            summary[datatype]["recovery"][pdb] = recoveries

            # Save uniqueness
            # Uniqueness = number of unique designs / total number of designs
            summary[datatype]["uniqueness"][pdb] = [
                len(list(set(summary[datatype]["design"][pdb]))) / NUM_SAMPLES
            ]

            # Save frequency per position
            summary[datatype]["frequency"][pdb] = get_frequency_of_residues(
                summary[datatype]["design"][pdb], NUM_SAMPLES
            )

    # Convert to pandas DataFrame
    for datatype in ["pMHC", "no_pMHC"]:
        # Convert designs to pandas DataFrame
        samples = pd.DataFrame(summary[datatype]["design"])
        samples.to_csv(os.path.join(basedir, datatype, "designs.csv"))

        # Convert recoveries to pandas DataFrame
        recoveries = pd.DataFrame(summary[datatype]["recovery"])
        recoveries.to_csv(os.path.join(basedir, datatype, "recoveries.csv"))

        # Convert uniqueness to pandas DataFrame
        uniqueness = pd.DataFrame(summary[datatype]["uniqueness"])
        uniqueness.to_csv(os.path.join(basedir, datatype, "uniqueness.csv"))

        # Convert frequency to pandas DataFrame
        frequency = pd.DataFrame(summary[datatype]["frequency"])
        frequency.to_csv(os.path.join(basedir, datatype, "frequency.csv"))


def testing_temperature(
    model: GVPTransformerModel, alphabet: Alphabet, temperatures: List[float]
):
    print("====== Temperature ======\n")
    # Read configuration file
    config = read_config("temperature.json")

    # Create directory
    os.makedirs(os.path.join("results", "temperatures"), exist_ok=True)

    for temperature in temperatures:
        print(f"\n=== {temperature} ===\n")
        os.makedirs(
            os.path.join("results", "temperatures", f"{temperature}"), exist_ok=True
        )

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
            basedir = os.path.join("results", "temperatures", f"{temperature}")
            pdbfile = os.path.join("data", "dataset", f"{pdb}.pdb")
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
                temperature,
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


def testing_sampling(
    model: GVPTransformerModel, alphabet: Alphabet, num_samples: List[int]
):
    print("====== Sampling ======\n")
    # Read configuration file
    config = read_config("sampling.json")

    # Create directory
    os.makedirs(os.path.join("results", "sampling"), exist_ok=True)

    for num_sample in num_samples:
        print(f"\n=== {num_sample} ===\n")
        os.makedirs(os.path.join("results", "sampling", f"{num_sample}"), exist_ok=True)

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
            basedir = os.path.join("results", "sampling", f"{num_sample}")
            pdbfile = os.path.join("data", "dataset", f"{pdb}.pdb")
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
                num_sample,
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
                len(list(set(summary["design"][pdb]))) / num_sample
            ]

            # Save frequency per position
            summary["frequency"][pdb] = get_frequency_of_residues(
                summary["design"][pdb], num_sample
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


if __name__ == "__main__":
    # Load model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()

    # Testing on 6ZKW
    testing_6ZKW(model, alphabet)

    # Testing on pMHC interference on protein sequence design
    testing_pMHC(model, alphabet)

    # Testing temperature based on sequence recovery, uniqueness and frequency
    testing_temperature(
        model,
        alphabet,
        temperatures=[
            5,
            2,
            1,  # 1e0
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,  # 1e-1
            0.01,  # 1e-2
            0.001,  # 1e-3
            0.0001,  # 1e-4
            0.00001,  # 1e-5
            0.000001,  # 1e-6
        ],
    )

    # Testing sampling based on sequence recovery, uniqueness and frequency
    testing_sampling(
        model,
        alphabet,
        num_samples=[5, 10, 25, 50, 100, 250, 500],
    )
