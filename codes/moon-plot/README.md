## Plotting for moons dataset

Plots can be obtained by running `generate_plots.py`, by altering the flags as necessary. Following are some important flags:

+ `model`: Can be one of `GI`, `baseline`, `grad_reg`, `inc_finetune`, `cida`, `cdot`
+ `epoch_classifier` (Integer): Number of pre-training epochs (Default: `30`)
+ `epoch_finetune` (Integer): Number of finetuning epochs (Default: `25`)
+ `seed` (Integer): Random seed
+ `preprocess`: Whether to pre-process the data (should be set for first run)

Example:

`python3 generate_plots.py --model GI --seed 1`
