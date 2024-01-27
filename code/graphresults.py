import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from ioutils import load_df_from_file, filenames_in_folder


def plot_submission_results(dataset_name):
    plt.style.use(["seaborn-v0_8-whitegrid"])
    import matplotlib.pyplot as mpl
    mpl.rcParams['font.size'] = 10

    task1_file_dir = "../score/task1/"
    task2_file_dir = "../score/task2/"
    task1_file_names = filenames_in_folder(task1_file_dir)

    dataset_type = dataset_name[-1]
    type_name = "Val"
    save_name = f"{type_name}C_{dataset_type}"

    def get_task1_scores_from_txt(file_dir: str, file_names: list[str]) -> pd.DataFrame:
        score_data = [load_df_from_file(f"{file_dir}{file_name}") for file_name in file_names if
                      f"T1{dataset_name[-1].lower()}" in file_name]
        score_df = pd.concat(score_data)
        score_df[0] = ["-".join(name.removesuffix(".txt").split("_")[-2:]) if "minVal" in name else
                       name.removesuffix(".txt").split("_")[-1] for name in score_df[0]]
        score_df.columns = ["name", "lc", "c", "hc", "count"]
        score_df = score_df.sort_values(by="c", ascending=True)

        name_map = {'SurvTRACE-minVal': "SurvTRACE - MinVal", 'survRF': "Random Forest", 'CGBSA': "CGBSA",
                    'survRFmri': "Random Forest MRI", 'survGB': "Gradient Boosting",
                    "AvgEnsemble": 'Ensemble Avg.', 'AvgEnsemble-minVal': 'Ensemble Avg. - MinVal',
                    'survGB-minVal': 'Gradient Boosting - MinVal', 'SurvTRACE': "SurvTRACE"}
        score_df["name"].replace(name_map, inplace=True)
        return score_df

    def get_task2_scores(file_dir: str, dataset_name) -> pd.DataFrame:
        file_names = filenames_in_folder(file_dir)
        score_data = [load_df_from_file(f"{file_dir}{file_name}") for file_name in file_names if
                      f"T2{dataset_name[-1].lower()}" in file_name]
        score_df = pd.concat(score_data)
        score_df[0] = ["-".join(name.removesuffix(".txt").split("_")[-2:]) if "minVal" in name else
                       name.removesuffix(".txt").split("_")[-1] for name in score_df[0]]

        auroc_cols = [f"{'L' if i % 3 == 0 else '' if i % 3 == 1 else 'H'}AUROC (0-{(i // 3) * 2 + 2})" for i in
                      range(15)]
        oe_cols = [f"{'L' if i % 3 == 0 else '' if i % 3 == 1 else 'H'}O/E (0-{(i // 3) * 2 + 2})" for i in range(15)]
        col_names = ["name", *auroc_cols, *oe_cols, "count"]
        score_df.columns = col_names
        score_df = score_df.iloc[score_df.filter(regex=f"^{'AUROC'}").mean(axis=1).argsort()]  # sort by avg auroc
        score_df = score_df.iloc[::-1]

        name_map = {'SurvTRACE-minVal': "SurvTRACE - MinVal", 'survRF': "Random Forest", 'CGBSA': "CGBSA",
                    'survRFmri': "Random Forest MRI", 'survGB': "Gradient Boosting",
                    "AvgEnsemble": 'Ensemble Avg.', 'AvgEnsemble-minVal': 'Ensemble Avg. - MinVal',
                    'survGB-minVal': 'Gradient Boosting - MinVal', 'SurvTRACE': "SurvTRACE"}
        score_df["name"].replace(name_map, inplace=True)

        return score_df

    def make_spider(df: pd.DataFrame, yticks, save_loc, ytick_bold: int = None, legend=True):
        import matplotlib.colors as mcolors

        colors = [*mcolors.TABLEAU_COLORS]
        categories = df.columns.values
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]

        plt.rc('figure', figsize=(10, 8))
        ax = plt.subplot(1, 1, 1, polar=True)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles, categories, color='black', size=18, weight='bold')
        ax.tick_params(axis='x', rotation=5.5)

        ax.set_rlabel_position(0)
        plt.yticks(yticks, color="black", size=18)
        y_lim_low_val = yticks[0]
        plt.ylim(y_lim_low_val, yticks[-1] + (yticks[-1] - yticks[-2]))

        if ytick_bold:
            labels = ax.get_yticklabels()
            labels[ytick_bold].set_fontweight('bold')
            labels[ytick_bold].set_fontsize(20)

            gridlines = ax.yaxis.get_gridlines()
            gridlines[ytick_bold].set_color("k")
            gridlines[ytick_bold].set_linewidth(3)

        angles = [0, *angles, 0]

        for i, (name, values) in enumerate(df.iterrows()):
            values = values.to_list()
            values = [values[0], *values, values[0]]
            if i == -2:
                ax.plot(angles, values, color=colors[i], linewidth=5, linestyle='solid')
                ax.fill(angles, values, color=colors[i], alpha=0.5, label='_nolegend_', hatch="x")
            else:
                ax.plot(angles, values, color=colors[i], linewidth=3, linestyle='solid')
                ax.fill(angles, values, color=colors[i], alpha=0.2, label='_nolegend_')

        if legend:
            ax.legend(df.index.to_list(), loc='upper left', bbox_to_anchor=(-0.2, 1),
                      ncol=1, fancybox=True, shadow=True, fontsize=20,
                      frameon=True, framealpha=1, facecolor="white", edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{save_loc}.png")
        plt.savefig(f"{save_loc}.pdf")
        plt.show()

    def plot_task2(file_dir):
        scores = get_task2_scores(file_dir)

        scores_auroc = scores.set_index("name").iloc[:, 1:16:3]
        t2A_ticks = [0.7, 0.75, 0.8, 0.85, 0.9, ]
        t2B_ticks = [0.4, 0.45, 0.5, 0.55, 0.6, ]
        make_spider(scores_auroc, t2A_ticks, save_loc=f"../graphs/t2{dataset_type}auroc",
                    # ytick_bold=2
                    )

        scores_oe = scores.set_index("name").iloc[:, 16:-1:3]
        t2A_ticks_oe = [0.5, 1, 1.5, 2, 2.5]
        t2B_ticks_oe = [0, 0.5, 1, 1.5, 2, ]
        make_spider(
            scores_oe,
            t2A_ticks_oe,
            save_loc=f"../graphs/t2{dataset_type}oe",
            ytick_bold=1,
            legend=False
        )

    def plot_multi_C(file_dir, file_names):
        test_scores_df = get_task1_scores_from_txt(file_dir, file_names)
        test_scores_df = test_scores_df[["name", "lc", "c", "hc"]]
        test_scores_df["type"] = "Test"

        histories = get_histories()
        values_df = {name: history.get(f"{type_name} C-Score") for name, history in histories}

        values_df = pd.DataFrame(values_df)
        average_values = values_df.mean(axis=0)
        confidence_intervals = values_df.quantile(q=[0.025, 0.975])
        wandb_scores_df = pd.concat([average_values, confidence_intervals.transpose()], axis=1)
        wandb_scores_df = wandb_scores_df.clip(upper=0.995)
        wandb_scores_df = wandb_scores_df.reset_index()
        wandb_scores_df = wandb_scores_df.rename(columns={0.0: "c", 0.025: "lc", 0.975: "hc", "index": "name"})
        wandb_scores_df["type"] = "Val"

        wandb_scores_df["name_ordered"] = pd.Categorical(wandb_scores_df["name"], categories=test_scores_df["name"],
                                                         ordered=True)
        wandb_scores_df = wandb_scores_df.sort_values("name_ordered")

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.subplots_adjust(left=0.28)
        colors = ["tab:blue", "tab:orange"]
        for offset, score_df in enumerate([wandb_scores_df, test_scores_df]):

            y_pos = np.array(range(len(score_df["name"])))

            for i, interval in enumerate(score_df[["lc", "hc"]].values):
                xerr = (interval[1] - interval[0]) / 2
                if score_df["c"].iloc[i] + xerr >= 0.995:
                    xerr = 0.995 - score_df["c"].iloc[i]

                ax.errorbar(
                    score_df["c"].iloc[i], y_pos[i] - 0.1 + offset * 0.2, xerr=xerr, elinewidth=4,
                    capsize=10, linewidth=4, markersize=10, color=colors[offset],
                )
            ax.scatter(score_df["c"], y_pos - 0.1 + offset * 0.2, marker="D", color=colors[offset], linewidth=8)

        ax.set_xlabel('C-Index', fontweight='bold', fontsize=22)
        ax.set_ylabel('Submitted run', fontweight='bold', fontsize=22)
        ax.set_xlim((0, 1))

        plt.xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1], fontsize=20)

        labels = ax.get_xticklabels()
        labels[3].set_fontweight('bold')
        labels[3].set_fontsize(20)

        gridlines = ax.xaxis.get_gridlines()
        gridlines[3].set_color("k")
        gridlines[3].set_linewidth(3)
        gridlines[3].set_linestyle("--")

        plt.yticks(fontsize=20, ticks=y_pos, labels=list(score_df["name"]))
        ax.legend(["Val", "Test"], loc='lower left',
                  ncol=1, fancybox=True, shadow=True, fontsize=20,
                  frameon=True, framealpha=1, facecolor="white", edgecolor="black")
        # plt.grid()
        plt.tight_layout()
        plt.savefig(f"../graphs/CIndex_{dataset_type}comparison.png")
        plt.savefig(f"../graphs/CIndex_{dataset_type}comparison.pdf")
        plt.show()

    def get_histories(wandb_project):
        datasetA_runs = ["0pqogcd9", "pefhr65z", "bmxri49g", "eg47otd5", "tyke6gd4", "9ckubkgv", "i02w78l0",
                         "tcuu3nlg", "gpibc3q4"]
        datasetB_runs = ["aksefkpg", "dd6u6bby", "4j1qq5bt", "o1dl9vrp", "doh6ghxb", "38z0p36o", "3sojqzwl",
                         "25vl5d6k", "xqqj2y75"]

        current_dataset_runs = datasetA_runs if dataset_type == "A" else datasetB_runs

        api = wandb.Api()
        entity, project = 'mrhanzl', wandb_project
        runs = api.runs(entity + "/" + project)

        histories = []
        for run in runs:
            if run.id not in current_dataset_runs:
                continue
            history = run.history()
            histories.append((run.name, history))

        def sort_f(hist_entry):
            return hist_entry[1][f"{type_name} C-Score Average"].iloc[-1]

        histories.sort(key=sort_f, reverse=True)

        return histories

    # plot_CIndex(file_names)
    # plot_wandb_box()
    # plot_multi_C(task1_file_dir, task1_file_names)
    plot_task2(task2_file_dir)
