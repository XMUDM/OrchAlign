DATA_ROOT=${1}
Cell_Type=${2:-GM12878} 
if [ -z "$DATA_ROOT" ]; then
    echo "Error: DATA_ROOT is not provided."
    exit 1
fi

python -m train \
    experiment=hg38/gene_express_ali \
    wandb.mode=online \
    wandb.group="CAGE_${Cell_Type}_bimambaAli" \
    wandb.name=test \
    trainer.gpus="[6]" \
    hydra.run.dir="./outputs/gene_exp_CAGE_${Cell_Type}_bimambaRNP/test" \
    train.single_CV=11 \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
    model="gene_express_bimamba_alignment" \
    task="extract_rationale" \
    task.loss.kl_loss_weight=0.01 \
    model.config.prior_scale_factor=10.0 \
    model.config.marginal_mean=0.1 \
    model.config.beta_min=1 \
    dataset.data_folder="$DATA_ROOT"
