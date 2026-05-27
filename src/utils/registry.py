"""Class registry for models, layers, optimizers, and schedulers.

"""

optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine_warmup_timm": "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    "EPInformer": "src.models.sequence.EPInformer.EPInformer_v2",
    "GeneExpBiMamba": "src.models.sequence.GeneExpformer.GeneExpBiMamba",
    "GeneExpMamba": "src.models.sequence.GeneExpformer.GeneExpMamba",
    "GeneExpHyena": "src.models.sequence.GeneExpformer.GeneExpHyena",
    "GeneExpTransformer": "src.models.sequence.GeneExpformer.GeneExpTransformer",
    "GeneExpTransformerMamba": "src.models.sequence.GeneExpformer.GeneExpTransformerMamba",
    "GeneExpMambaCross": "src.models.sequence.GeneExpformer.GeneExpMambaCross",
    "BiMambaMIRNP": "src.models.sequence.GeneExpformer.GeneBiMambaMIRNP",
    "Enformer": "src.models.sequence.Enformer.GeneEnformer",
    "BiMambaAlign": "src.models.sequence.GeneExpformer.GeneExpAlignment",
    "BiMambaAlignWoEncoder": "src.models.sequence.GeneExpformer.GeneExpAlignmentWoEncoder",
    
}

layer = {
    "hyena": "src.models.sequence.hyena.HyenaOperator",
    "hyena-filter": "src.models.sequence.hyena.HyenaFilter",
}

callbacks = {
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "model_checkpoint_every_n_steps": "pytorch_lightning.callbacks.ModelCheckpoint",
    "model_checkpoint_every_epoch": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "params": "src.callbacks.params.ParamsLog",
    "timer": "src.callbacks.timer.Timer",
    "grad_norm_monitor": "src.callbacks.grad_norm_monitor.GradNormMonitor",
    "val_every_n_global_steps": "src.callbacks.validation.ValEveryNGlobalSteps",
}

model_state_hook = {
    'load_backbone': 'src.models.sequence.dna_embedding.load_backbone',
}
