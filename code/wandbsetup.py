import wandb


def setup_wandb(config):
    wandb.login()
    run = wandb.init(
        project="IDPP-CLEF",
        config=config
    )

    return run