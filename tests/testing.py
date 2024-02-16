import sys
import warnings

import pandas as pd

sys.path.append("../")

from TCRDesign import *

# We are evaluating ESM-IF model on multiple chains of a TCR-pMHC complex.
# 6ZKW.pdb is a TCR-pMHC complex.
# 6ZKW_DE.pdb is only the TCR (D and E).
# 6ZKW_mut.pdb is a TCR-pMHC complex with mutations on the TCR interface residues to glycine.
# 6ZKW_mut.pdb is a TCR-pMHC complex with mutations on the residues surrounding TCR interface to glycine.

# Just suppress all warnings with this:
warnings.filterwarnings("ignore")

# SET YOUR PARAMETERS HERE
pdbs = ["6ZKW.pdb", "6ZKW_DE.pdb", "6ZKW_mut.pdb", "6ZKW_mut2.pdb"]
chains = ["D", "E"]
# 110,111,112,134,135 (chain D)
# 113,114,133 (chain E)
design = ["110D", "111D", "112D", "134D", "135D", "113E", "114E", "133E"]
num_samples = 10
temperature = 1.0
verbose = True

if __name__ == "__main__":
    # Load model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()
    model = model.cuda()

    # Summary of recoveries
    summary = {}

    # Sampling sequences in pdbs
    for pdbfile in pdbs:
        outpath = f"results/{pdbfile.split('.')[0]}.fasta"

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
