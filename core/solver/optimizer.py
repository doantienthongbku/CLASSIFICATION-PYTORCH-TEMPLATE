import torch


def initialize_optimizer(cfg, model):
    # args global
    optimizer_strategy = cfg.solver.optimizer
    learning_rate = cfg.solver.learning_rate
    weight_decay = cfg.solver.weight_decay
    
    # args for SGD
    SGD_momentum = cfg.optimizer_args.SGD.momentum
    SGD_nesterov = cfg.optimizer_args.SGD.nesterov
    
    # args for Adam and AdamW
    ADAM_betas1 = cfg.optimizer_args.ADAM.betas1
    ADAM_betas2 = cfg.optimizer_args.ADAM.betas2
    ADAM_eps = float(cfg.optimizer_args.ADAM.eps)
    ADAM_amsgrad = cfg.optimizer_args.ADAM.amsgrad
    
    # args for AdaGrad
    ADA_lr_decay = cfg.optimizer_args.ADAGRAD.lr_decay
    ADA_initial_accumulator_value = cfg.optimizer_args.ADAGRAD.initial_accumulator_value
    ADA_eps = float(cfg.optimizer_args.ADAGRAD.eps)
    
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=SGD_momentum,
            nesterov=SGD_nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            batas=(ADAM_betas1, ADAM_betas2),
            eps=ADAM_eps,
            amsgrad=ADAM_amsgrad
        )
    elif optimizer_strategy == 'ADAMW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            batas=(ADAM_betas1, ADAM_betas2),
            eps=ADAM_eps,
            amsgrad=ADAM_amsgrad
        )
    elif optimizer_strategy == 'ADAGRAD':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            lr_decay=ADA_lr_decay,
            initial_accumulator_value=ADA_initial_accumulator_value,
            eps=ADA_eps
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer
