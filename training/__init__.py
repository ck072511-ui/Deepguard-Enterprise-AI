"""
DeepGuard — Training package.

Encapsulates all training infrastructure: training loops, callbacks,
learning rate schedulers, loss functions, and optimizers.

Packages:
    training.callbacks   — Early stopping, checkpointing, LR monitoring
    training.schedulers  — Cosine annealing, warmup, OneCycleLR wrappers
    training.losses      — Custom loss functions (focal, label smoothing)
    training.optimizers  — Optimizer factories and SAM optimizer
    training.evaluators  — Evaluation metric computation
"""
