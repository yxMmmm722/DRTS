# DRTS

This codebase is built on the [Time-Series-Library](https://github.com/thuml/Time-Series-Library/tree/main). All datasets used by DRTS can be downloaded from that repository.

## Requirements

Install the dependencies.

```bash
pip install -r requirements.txt
```

## Usage

1. Create the `./data` directory and place the dataset files in the corresponding subdirectories.
2. Run the code.

```bash
python run.py --data [DATASET]
```

For example:

```bash
python run.py --data ETTh1 
```
