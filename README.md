# SCER
Code Implementation for Spurious Correlation Aware Embedding Regularization for Worst Group Robustness

## Code Implementation

## Installation

First, clone our repository and set up the environment:

```bash
# Clone our repository first
# Create and activate the conda environment
conda env create -f environment

```

To download the required data, run the following command:

```
python -m scer.scripts.download --data_path <data_path> --download
```

## Training

To train SCER for one time, use the following command:

```bash
python -m scer.train --algorithm SCER --dataset "Your Data" --train_attr yes \
    --data_dir "Your_path" --output_dir "Your_path"
```
To run a sweep for SCER, use the following command:
```
python -m scer.sweep launch --algorithms SCER --dataset "Your Data" \
    --train_attr no --n_hparams "Your Num" --n_trials 1
```
## Collecting Results

After training, you can collect and summarize the results using the following command:

```bash
python -m scer.collect_results --input_dir "Your output path" 
```

## Acknowledgements
This implementation is primarily based on SubpopBench ([GitHub Repository](https://github.com/YyzHarry/SubpopBench)).  
We would like to acknowledge and thank the authors of SubpopBench for their work and contributions.
