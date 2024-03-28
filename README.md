# ESMIFDesign

This repository focuses on designing T-cell receptors (TCRs) using the [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding) deep learning method.

The ESM-IF1 inverse folding method is built for predicting protein sequences from their backbone atom coordinates. Here, we use the ESM-IF1 model (esm_if1_gvp4_t16_142M_UR50 - fair-esm v2.0.1) to design part of the TCRs sequences, considering their peptide-major histocompatibility complex (pMHC) complex. Positions outside the specified criteria were held constant. The designed positions were changed to <mask> tokens, and all amino acid substitutions, including cysteine, were allowed. Given the multi-chain structure, a padding of 10 <pad> tokens was used to separate the chains.

## Dependencies

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

The configuration file is located at `config.json`, where it specifies, for each structure, the residues and chains to be designed. The file is organized as follows:

```json
{
    "6zkw": ["110D","111D","112D","134D","135D","113E","114E","133E"],
    ...,
    "8shi": ["109D","110D","111D","112D","113D","114D","135D","110E","111E","112E","113E","134E","135E"]
}
```

To design TCR sequences, run:

```bash
python run.py
```

## Testing

We tested some conditions to check the performance of the model.

1. Testing on `6ZKW` structure:
    - `6ZKW.pdb` represents a TCR-pMHC complex.
    - `6ZKW_DE.pdb` includes only the TCR (D and E chains).
    - `6ZKW_contact_to_gly.pdb` is a TCR-pMHC complex with mutations on the TCR residues in contact with the pMHC, changing them to glycine.
    - `6ZKW_all_gly.pdb` is a TCR-pMHC complex with mutations on all residues, changing them to glycine.
2. Testing on pMHC interference on protein sequence design (`tests/pMHC1.config`)
3. Testing temperature in the interval (`tests/temperature.config`): [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 5]
4. Testing number of samples in the interval (`tests/sampling.config`): [5, 10, 25, 50, 100, 250, 500]
5. Testing design approaches:
    - `CDR3 interface` (`tests/CDR3_interface.config`): restricting the design to CDR3 (α and β TCR chains) within a proximity of 5 Å to either the peptide or MHC;
    - `CDR3` (`tests/CDR3.config`): designing the entire CDR3 (α and β TCR chains).
    - `CDRs interface` (`tests/CDRs_interface.config`): designing CDR1, CDR2, or CDR3 positions (α and β TCR chains) within a 5 Å distance to the peptide or MHC;

To run the tests, navigate to the tests directory and execute:

```bash
cd tests
python testing.py
```
