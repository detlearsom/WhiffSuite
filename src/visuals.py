import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import metrics

def saveHist(_file, data, bins=100):
    plt.hist(data, bins)
    plt.savefig(_file)


def mutInfoGraph(results: dict, name: str):
    for attack in results.keys():
        x = []
        y = []
        i = 0
        for (_, item) in results[attack].items():
            x.append(i)
            y.append(item)
            i += 1
        plt.plot(x, y, label=attack)
    plt.legend()
    if name != None:
        with open("mutinfo.json", "w+") as f:
            json.dump(results, f)
        plt.savefig(name)


def mutInfoGraphPipeline(data, metadata, name):
    results = {}
    targets = data[metadata['label_field']].value_counts().index.tolist()
    for target in targets:
        if target == metadata['benign_label']:
            continue
        drop_size = data[data[metadata['label_field']] == metadata['benign_label']
                         ].shape[0] - data[data[metadata['label_field']] == target].shape[0]
        drop = data[data[metadata["label_field"]] ==
                    metadata["benign_label"]].sample(n=drop_size)
        df = data.drop(drop.index)
        gains = metrics.info_gain(df, metadata, target)
        results[target] = {k: v for k, v in sorted(
            gains.items(), key=lambda item: item[1])}
    mutInfoGraph(results, name)

def cosSimGraph(results, cutoffs, _name):
    value_list = []
    dataset_col = []

    for key, item in results.items():
        value_list.append(item)
        dataset_col.append('CosSim')
        value_list.append(cutoffs[key])
        dataset_col.append('CosSimCutoff')

    df = pd.DataFrame(data={"Type": dataset_col, "Values": value_list})

    ax = sns.violinplot(x="Type", y="Values", data=df,
                        inner=None, color=".92")
    ax = sns.stripplot(x="Type", y="Values", hue="Type",
                       data=df, palette="viridis", size=8,
                       marker="d", edgecolor="gray", alpha=.75)
    ax.set(yscale="linear")
    ax.get_legend().remove()
    plt.savefig(_name)


def overlay_multiple_kdes(data, x, y, metadata, targets, color_scheme, levels):
    label_field = metadata["label_field"]

    colors = sns.color_palette(color_scheme, len(targets))
    #graph = sns.jointplot(data=data[data[label_field] == benign_label], x=x, y=y, kind='kde', color=colors[0])
    graph = sns.jointplot()
    #labels.remove(benign_label)
    patch_list = []
    #patch_list.append(mpatches.Patch(color=colors[0], label=benign_label))
    
    
    for idx, label in enumerate(targets):
        graph.x = data[data[label_field] == label].sample(25)[x]
        graph.y = data[data[label_field] == label].sample(25)[y]
        
        if (np.std(graph.x) < 0.05) or (np.std(graph.y) < 0.05):
            graph.plot_joint(sns.scatterplot, color=colors[idx], s=80)
            graph.plot_marginals(sns.rugplot, color=colors[idx], height=0.1, clip_on=False)
        else:
            graph.plot_joint(sns.kdeplot, color=colors[idx], levels=levels)
            sns.kdeplot(
            x=graph.x,
            ax=graph.ax_marg_x,
            color=colors[idx])

            sns.kdeplot(
            y=graph.y,
            ax=graph.ax_marg_y,
            color=colors[idx])
        patch_list.append(mpatches.Patch(color=colors[idx], label=label))
        
    plt.legend(handles=patch_list)


def correlation_heatmap(correlations, _file):
    fig, ax = plt.subplots(figsize=(10, 16))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.savefig(_file)

