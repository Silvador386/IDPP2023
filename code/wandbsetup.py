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

sweep_configuration = {
    "name": "GradientBoostingSweep",
    "metric": {"name": "Val C-Score", "goal": "maximize"},
    "method": "bayes",
    "parameters": {
        'n_estimators': {"values": [100, 300, 500]},
        'max_depth': {"values": [ 6, 8, 10]},
        'min_samples_split': {"values": [ 8, 10, 15]},
        'min_samples_leaf': {"values": [ 4, 6, 8]},
        # 'min_weight_fraction_leaf': {"values": [0.0, 0.3]},
        'max_features': {"values": ["sqrt", "log2"]},
        # 'max_leaf_nodes': None,
        # 'bootstrap': {"values": [True, False]},
        # 'oob_score': {"values": [False, True]},
        # 'warm_start': {"values": [False, True]},
        # 'max_samples': None
    }
}

gbs_sweep_params = {"loss": {"values": ['coxph', 'squared', 'ipcwls']},
                    "learning_rate": {"values": [0.1, 0.5, 1]},
                    'n_estimators': {"values": [100, 300, 500]},
                    # "criterion": {"values": ['mse', 'friedman_mse']},
                    'min_samples_split': {"values": [2, 4, 6, 8]},
                    'min_samples_leaf': {"values": [1, 3, 5]},
                    'min_weight_fraction_leaf': {"values": [0.0, 0.3]},
                    'max_depth': {"values": [3, 5, 7]},
                    "min_impurity_decrease": {"values": [0.0, 0.5, 1]},
                    'max_features': {"values": ["auto", "sqrt", "log2"]},
                    # max_leaf_nodes=None,
                    "subsample": {"values": [0.2, 0.5, 1]},
                    "dropout_rate": {"values": [0.0, 0.2, 0.5]},
                    "ccp_alpha": {"values": [0.0, 0.1, 1, 10]}}

cgbs_sweep_params = {"loss": {"values": ['coxph', 'squared', 'ipcwls']},
                     "learning_rate": {"values": [0.1, 0.5, 1]},
                     'n_estimators': {"values": [100, 200, 300, 500]},
                     "subsample": {"values": [0.2, 0.5, 1]},
                     "dropout_rate": {"values": [0.0, 0.2, 0.5]}
                     }


def launch_sweep(project, notes, config):
    sweep_configuration[notes] = notes
    sweep_configuration.update(config)
    sweep_configuration.update(config)
    sweep_id = wandb.sweep(sweep_configuration, project=project)
    return sweep_id
