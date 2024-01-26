import wandb
from dataset import IDDPDataset

def setup_wandb(config: dict, model_name: str, dataset: IDDPDataset, *kwargs):
    wandb.login()

    dataset_name = dataset.dataset_name

    project = f"IDPP-CLEF-{dataset_name[-1]}{'_V3' if dataset_name == 'datasetA' else ''}"

    notes = "(stat_vars[onehot])_(edss)_(delta_relapse_time0[funcs])_(evoked_potential[type][twosum])_final_avg"

    config["model"] = model_name
    run = wandb.init(
        project=project,
        config=config,
        reinit=True,
        name=model_name,
        notes=notes
    )

    return run


sweep_configuration = {
    "name": "SurfTraceSweep",
    "metric": {"name": "Val C-Score", "goal": "maximize"},
    "method": "random",
    "parameters": {
        'n_estimators': {"values": [100, 300, 500]},
        'max_depth': {"values": [6, 8, 10]},
        'min_samples_split': {"values": [8, 10, 15]},
        'min_samples_leaf': {"values": [4, 6, 8]},
        # 'min_weight_fraction_leaf': {"values": [0.0, 0.3]},
        'max_features': {"values": ["sqrt", "log2"]},
        # 'max_leaf_nodes': None,
        # 'bootstrap': {"values": [True, False]},
        # 'oob_score': {"values": [False, True]},
        # 'warm_start': {"values": [False, True]},
        # 'max_samples': None
    }
}

gbs_sweep_params = {"loss": {"values": ['coxph']},
                    "learning_rate": {"values": [0.1, 0.5]},
                    'n_estimators': {"values": [100, 200]},
                    # "criterion": {"values": ['mse', 'friedman_mse']},
                    'min_samples_split': {"values": [2, 4]},
                    'min_samples_leaf': {"values": [1]},
                    # 'min_weight_fraction_leaf': {"values": [0.0, 0.3]},
                    'max_depth': {"values": [3, 5, 7]},
                    # "min_impurity_decrease": {"values": [0.0, 0.5, 1]},
                    # 'max_features': {"values": ["sqrt", "log2"]},
                    # max_leaf_nodes=None,
                    "subsample": {"values": [0.2, 0.5, 1]},
                    "dropout_rate": {"values": [0.0, 0.2]},
                    "ccp_alpha": {"values": [0.0, 0.1, 1]}
                    }

cgbs_sweep_params = {"loss": {"values": ['coxph', 'squared', 'ipcwls']},
                     "learning_rate": {"values": [0.1, 0.5, 1]},
                     'n_estimators': {"values": [100, 200, 300, 500]},
                     "subsample": {"values": [0.2, 0.5, 1]},
                     "dropout_rate": {"values": [0.0, 0.2, 0.5]}
                     }

surftrace_sweep_params = {'batch_size': {"min": 48, "max": 128},
                          'weight_decay': {"min": 5e-5, "max": 1e-3},
                          'learning_rate': {"min": 5e-4, "max": 1e-2},
                          'epochs': {"values": [40]},
                          'hidden_size': {"values": [16, 32, 64]}, # embedding size
                          'intermediate_size': {"values": [ 64, 128, 256, 512]}, # intermediate layer size in transformer layer
                          'num_hidden_layers': {"values": [2, 4, 6]}, # num of transformers
                          'num_attention_heads': {"values": [2, 4, 6]}, # num of attention heads in transformer layer
                          'hidden_dropout_prob': {"min": 0.2, "max": 0.4},
                          'attention_probs_dropout_prob': {"min": 0.1, "max": 0.3},
                          }


def launch_sweep(project: str, notes: str, config: dict) -> str:
    sweep_configuration[notes] = notes
    sweep_configuration.update(config)
    sweep_configuration["parameters"] = surftrace_sweep_params
    sweep_id = wandb.sweep(sweep_configuration, project=project)
    return sweep_id
