# TCRDesign

Design TCRs using alternative deep learning methods (e.g., [ESMIF](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding)).

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

To execute analysis, run:

```bash
python run.py
```
