# OrchAlign

Our code is built on the source code provided by Seq2Exp (https://github.com/divelab/AIRS/)

## Installation

- Please refer to `mamba_env.yml` for the exact package versions. 
# Dataset

The dataset for this repo can be downloaded from https://huggingface.co/datasets/xingyusu/GeneExp. 
Set the dataset directory as `$DATA_ROOT` before running any experiments.

# Training and Evaluation

To reproduce the results of OrchAlign, run the following
```bash
sh Seq2ExpAli.sh $DATA_ROOT
```

# Model Class

The class file of Our model is  "GeneExpMambaCross" in  "src\models\sequence\GeneExpformer.py"
