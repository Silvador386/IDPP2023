import wandb


def setup_wandb(project, config, name=None, *kwargs):
    wandb.login()
    run = wandb.init(
        project=project,
        config=config,
        reinit=True,
        name=name,
        *kwargs
    )

    return run
