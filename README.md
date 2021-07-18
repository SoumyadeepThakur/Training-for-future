# Gradient Interpolation

This contains our implementation of the **Gradient Interpolation** and some other baselines.

## Running the code

All code files are present in `codes/GI`

All experiments can be run from `main.py`, by altering the flags as necessary. Following are some important flags:

+ `data`: Can be one of `moons`, `mnist`, `house`, `m5`, `m5_household`, `onp`, `elec2`
+ `model`: Can be one of `GI`, `baseline`, `grad_reg`, `inc_finetune`, `goodfellow`, `t_incfinetune`, `t_goodfellow`
+ `epoch_classifier` (Integer): Number of pre-training epochs
+ `epoch_finetune` (Integer): Number of finetuning epochs
+ `seed` (Integer): Random seed
+ `preprocess`: Whether to pre-process the data (should be set for first run)

Example:

Running GI on 2-moons dataset for `35` pre-training epochs and `25` fine-tuning epochs

`python3 main.py --data moons --model GI --train_algo grad_int --preprocess --epoch_classifier 35 --epoch_finetune 25 --seed 1`
