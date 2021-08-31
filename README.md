# General Usage

Most arguments are specified in the config files of `config/`, you can edit entries in these or pass paths as command line arguments to match to however your local machine is set up.

## Embeddings

For `main*py` scripts, arguments are primarily handled by the config files. The only neccessary command-line arguments to replicate experiments are 
```
dataset.task_id=`seq -s, 0 n` -m
```
where `n` is 24 for cub-only, 29 for cars, or 49 for cub-inat. The `seq` expands out into a comma-separated range and the `-m` flag tells the script to execute in multi-run mode.

LEEP/RSA/Cui et al.'s EMD do not require the task_id flag as the iteration is done automatically not using the `hydra` module. These can be run from the corresponding `embed*py`.




## Evaluation

For `accuracy_embeddings*py` scripts, the primary argument is the location of the embeddings file. This is typically given as `--root [path to file dir]`.
Note that the CUB Attribute evaluation requires *two* roots as attributes are only defined for CUB so the iNat embeddings from CUB-iNat are typically re-used.

LEEP/RSA/Cui et al.'s EMD have a hardcoded path that their corresponding `embed` script saves to (these are for cub-inat only). The TTK accuracy scripts have hardcoded paths to the example similarity matrices.(TODO: Add w/ TTK instructions)

## Training/Transferring Experts

To train new experts use `train_experts.py`, task_ids are as above. To train on CUB *attributes* use `dataset.name=cub_attributes` and `dataset.attribute=[desired attribute]` (e.g. `breast_pattern`). To train car experts use `dataset.name=cars`.

To transfer experts among each other use `transfer_experts.py` (with attribute differences as before) and `transfer_experts_cars.py` 


## Other scripts

`print_pseudolabel_preds.py` is run similarly to `main*py` scripts to display the pseudolabels versus actual class index for a given task.


# Environment

Conda environment is in environment.yaml, run `conda env create -f environment.yml`




# Data

   Cars: https://drive.google.com/file/d/1deqk9Ep3yf7ejwXV3mgQ1A6ZWIf5XyRS/view?usp=sharing
   
   CUB: https://drive.google.com/file/d/1IwaFpQnbANlSjZNZ1NLZg7MxRcUblUhH/view?usp=sharing
   
   iNat (info only): https://drive.google.com/file/d/1KqNc52ApY4axccVWdDOMBGL582WWzQsB/view?usp=sharing
   
   iNat (images): https://github.com/visipedia/inat_comp/blob/master/2018/README.md#Data (train and val images)



# Results
Embeddings are available at:
https://drive.google.com/file/d/1DkQQfQpvSLxv7p6C1n_vzXjth8jasyQh/view?usp=sharing

Note that there is slight variation between the numbers produced by the above embeddings and our paper.
