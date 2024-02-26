import os
import sys
import warnings

import pandas as pd

sys.path.append("../")

from TCRDesign import *

# Just suppress all warnings with this:
warnings.filterwarnings("ignore")

# We are assessing the ESM-IF model on multiple chains of a TCR-pMHC complex.
# - 6ZKW.pdb represents a TCR-pMHC complex.
# - 6ZKW_DE.pdb includes only the TCR (D and E chains).
# - 6ZKW_contact_to_gly.pdb is a TCR-pMHC complex with mutations on the TCR
# residues in contact with the pMHC, changing them to glycine.
# - 6ZKW_all_gly.pdb is a TCR-pMHC complex with mutations on all residues,
# changing them to glycine.

# WARNING: The following code does not correctly calculate the recoveries for
# the 6ZKW_contact_to_gly.pdb and 6ZKW_all_gly.pdb files. This is because the
# native sequences are read from the pdb files.

# SET YOUR PARAMETERS HERE
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
num_samples = 10
temperature = 1.0
padding_length = 10
verbose = True

# %%

if __name__ == "__main__":
    # Load model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()

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
            num_samples,
            temperature,
            padding_length,
            verbose,
        )

        # Append recoveries to summary
        recoveries.append(sum(recoveries) / len(recoveries))
        summary[pdbfile] = recoveries

    # Print summary
    summary = pd.DataFrame(summary)
    summary.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Average"]
    summary.to_csv("results/summary.csv")
    print(summary)
