# Testing on 6ZKW

We are assessing the ESM-IF model's performance on multiple chains of a TCR-pMHC complex.

## Dataset

- `6ZKW.pdb` represents a TCR-pMHC complex.
- `6ZKW_DE.pdb` includes only the TCR (D and E chains).
- `6ZKW_contact_to_gly.pdb` is a TCR-pMHC complex with mutations on the TCR residues in contact with the pMHC, changing them to glycine.
- `6ZKW_all_gly.pdb` is a TCR-pMHC complex with mutations on all residues, changing them to glycine.

*NOTE*: The following code does not correctly calculate the recoveries for the 6ZKW_contact_to_gly.pdb and 6ZKW_all_gly.pdb files. This is because the native sequences are read from the pdb files.
