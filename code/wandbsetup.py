import wandb


def setup_wandb(project, config):
    wandb.login()
    run = wandb.init(
        project=project,
        config=config,
        reinit=True
    )

    return run