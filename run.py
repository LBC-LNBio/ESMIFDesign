import warnings

from TCRDesign.esmif import esm, sample_seq_multichain

# Just suppress all warnings with this:
warnings.filterwarnings("ignore")

# SET YOUR PARAMETERS HERE
pdbfile = "data/6ZKW.pdb"
outpath = "results/6ZKW.fasta"
chains = ["D", "E"]
# 110,111,112,134,135 (chain D)
# 113,114,133 (chain E)
design = ["110D", "111D", "112D", "134D", "135D", "113E", "114E", "133E"]
num_samples = 10
temperature = 1.0
verbose = True

if __name__ == "__main__":
    # UserWarning: Regression weights not found, predicting contacts will not produce correct results.
    # @tomsercu: You don't need the regression weights, these are for contact prediction only. They are not uploaded on purpose to prevent folks from inadvertently using esm-1v for contact prediction which will lead to poor results, as discussed in the paper.
    # https://github.com/facebookresearch/esm/issues/170#issuecomment-1076687163
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()
    model = model.cuda()

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
