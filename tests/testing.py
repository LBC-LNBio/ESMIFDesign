import os
import sys
import warnings

import pandas as pd

sys.path.append("../")

from TCRDesign import esm, get_chains, read_config, sample_seq_multichain

# Just suppress all warnings with this:
warnings.filterwarnings("ignore")

# CONSTANTS
NUM_SAMPLES = 10
TEMPERATURE = 1.0
PADDING = 10
VERBOSE = False


def testing_6ZKW(model, alphabet):
    # Define parameters
    pdbs = [
        "data/6ZKW.pdb",
        "data/6ZKW_DE.pdb",
        "data/6ZKW_contact_to_gly.pdb",
        "data/6ZKW_all_gly.pdb",
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
        outpath = f"results/{os.path.basename(pdbfile).replace('.pdb', '.fasta')}"

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
    summary.to_csv("results/6ZKW/summary.csv")
    print(summary)


def testing_pMHC(model, alphabet):
    # Read configuration file
    config = read_config("tests.json")

    # Create directory
    os.makedirs("results/pMHC_interference", exist_ok=True)
    os.makedirs("results/pMHC_interference/pMHC", exist_ok=True)
    os.makedirs("results/pMHC_interference/no_pMHC", exist_ok=True)

    # Create summary
    summary = {}
    summary["pMHC"] = {}
    summary["no_pMHC"] = {}

    for pdb in config:
        print(f"[==> {pdb}")

        for datatype in ["pMHC", "no_pMHC"]:
            print(f"> {datatype}")

            # Prepare parameters
            pdbfile = os.path.join("data", "pMHC_interference", datatype, f"{pdb}.pdb")
            outpath = os.path.join(
                "results", "pMHC_interference", datatype, f"{pdb}.fasta"
            )
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

    # Convert to pandas DataFrame
    for datatype in ["pMHC", "no_pMHC"]:
        summary[datatype] = pd.DataFrame(summary[datatype])
        summary[datatype].to_csv(
            os.path.join("data", "pMHC_interference", datatype, "summary.csv")
        )
        print(summary[datatype])


if __name__ == "__main__":
    # Load model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()

    # Testing on 6ZKW
    # testing_6ZKW(model, alphabet)

    # Testing on pMHC interference on protein sequence design
    testing_pMHC(model, alphabet)
