
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plotting tools 
import colorcet as cc
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
    
    
def get_embedding(data):
    """ Function to compute the UMAP embedding"""
    data_scaled = StandardScaler().fit_transform(data)

    embedding = umap.UMAP(n_neighbors=10,
                            min_dist=0.5,
                            metric='correlation',
                            random_state=16).fit_transform(data_scaled)
    return embedding


def draw_umap(embedding, label):
    contour_c = '#444444'
    labelsize = 25
    label = label
    palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(label)))

    fig, ax = plt.subplots(figsize=(20, 15))

    plt.xlim([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 1.5])
    plt.ylim([np.min(embedding[:, 1]) - 0.5, np.max(embedding[:, 1]) + 0.5])
    plt.xlabel('UMAP 1', fontsize=labelsize)
    plt.ylabel('UMAP 2', fontsize=labelsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # plt.scatter(embedding[:, 0], embedding[:, 1], lw=0, , label=label, alpha=1.0, s=180, linewidth=2)
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=label, 
                hue_order=list(np.unique(label)), palette=palette)
    leg = plt.legend(prop={'size': labelsize}, loc='upper right', markerscale=2.00)
    leg.get_frame().set_alpha(0.9)
    plt.setp(ax, xticks=[], yticks=[])
    plt.show()
    
    
def tsne_plot(embedding, label, pred, filename,seed):
    font2 = {'weight': 'bold',
                'size': 24,
                }
    font3 = {
        'weight': 'bold'
    }
    palette1 = sns.color_palette(cc.glasbey, n_colors=len(np.unique(label)))
    # palette2 = sns.color_palette(cc.glasbey, n_colors=len(np.unique(pred)))
    tsne_embedding = TSNE().fit_transform(embedding)
    # tsne_embedding = umap.UMAP().fit_transform(embedding)
    print("finish tsne process")
    # print(tsne_embedding.shape)
    # print(len(label))
    # print(len(pred))
    figsize = 20, 8
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(121)
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    # sns.scatterplot(tsne_embedding[:, 0], tsne_embedding[:, 1], hue=label, legend=False, palette=palette1)
    sns.scatterplot(tsne_embedding[:, 0], tsne_embedding[:, 1], hue=label, legend=False,
                    hue_order=list(np.unique(label)), palette=palette1)
    # plt.legend(bbox_to_anchor = (1., 1.))
    # plt.legend(loc='outside left upper')
    plt.tick_params(labelsize=24)
    plt.title("Ground-truth", font2)
    labels = ax1.get_xticklabels(font2) + ax1.get_yticklabels(font2)
    [label.set_fontname('Times New Roman') for label in labels]

    ax2 = plt.subplot(122)
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    # sns.scatterplot(tsne_embedding[:, 0], tsne_embedding[:, 1], hue=pred, legend=False, palette=palette2)
    sns.scatterplot(tsne_embedding[:, 0], tsne_embedding[:, 1], hue=pred, hue_order=list(np.unique(label)),
                    palette=palette1)
    # plt.legend(bbox_to_anchor=(1., 1.), title = "Cell type", prop = font3)
    plt.legend(bbox_to_anchor=(1., 1.), title="Cell type")
    # plt.legend(loc='outside right upper')
    plt.tick_params(labelsize=24)
    plt.title("Prediction", font2)
    labels = ax2.get_xticklabels(font2) + ax2.get_yticklabels(font2)
    [label.set_fontname('Times New Roman') for label in labels]
    plt.tight_layout()
    plt.savefig(filename+str(seed), dpi=300)
    plt.savefig(filename+str(seed).replace(".pdf", ".eps"), dpi=300)
    plt.savefig(filename+str(seed).replace(".pdf", ".png"), dpi=300)

# Cao_class_set = ['GABAergic neuron', 'cholinergic neuron', 'ciliated olfactory receptor neuron', 'coelomocyte',
#                  'epidermal cell',
#                  'germ line cell', 'glial cell', 'interneuron', 'muscle cell', 'nasopharyngeal epithelial cell',
#                  'neuron',
#                  'seam cell', 'sensory neuron', 'sheath cell', 'socket cell (sensu Nematoda)',
#                  'visceral muscle cell']
#
# Zeisel_2018_class_set = ['CNS neuron (sensu Vertebrata)', 'astrocyte', 'cerebellum neuron',
#                          'choroid plexus epithelial cell',
#                          'dentate gyrus of hippocampal formation granule cell',
#                          'endothelial cell of vascular tree', 'enteric neuron', 'ependymal cell', 'glial cell',
#                          'inhibitory interneuron',
#                          'microglial cell', 'neuroblast', 'oligodendrocyte', 'oligodendrocyte precursor cell',
#                          'peptidergic neuron', 'pericyte cell',
#                          'peripheral sensory neuron',
#                          'perivascular macrophage', 'radial glial cell', 'sympathetic noradrenergic neuron',
#                          'vascular associated smooth muscle cell']
#
# # file_list = os.listdir("case")
#
# # for typi in ["replay", "replay_and_proxy", "replay_and_proxy_and_uniform", "joint"]:
# for typi in ["replay_and_proxy_and_uniform"]:
#     # for typi in ["individual"]:
#     # for stage in [0]:
#     for stage in [0, 1, 2, 3]:
#         for name in ["Zeisel_2018"]:
#             # for name in ["Cao"]:
#             print("begin {} stage {} type {}".format(name, stage, typi))
#             sankey_df = pd.read_csv(
#                 "case/" + name + "_stage_" + str(stage) + "_test_data_" + typi + "_sankey_information.csv")
#             labels = np.array(sankey_df["true cell type"])
#             preds = np.array(sankey_df["pred cell type"])
#             feature_df = np.array(pd.read_csv(
#                 "case/" + name + "_stage_" + str(stage) + "_test_data_" + typi + "_visualization_feature.csv",
#                 index_col=0, header=0))
#             tsne_plot(feature_df, labels, preds, filename="case5/" + name + "_stage_" + str(
#                 stage) + "_test_data_" + typi + "_tsne_plot_1203.pdf")
#             print("finish {} stage {} type {}".format(name, stage, typi))

