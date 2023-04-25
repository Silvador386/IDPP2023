import wandb


def setup_wandb(project, config, name=None, notes=None, *kwargs):
    wandb.login()
    run = wandb.init(
        project=project,
        config=config,
        reinit=True,
        name=name,
        notes=notes
    )

    return run
