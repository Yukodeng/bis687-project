
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
# clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


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
                            metric='correlation').fit_transform(data_scaled)
    return embedding


def calculate_cluster_results(data, true_labels, seed):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    n_clusters = len(np.unique(true_labels))
    np.random.seed(seed)
    
    # get UMAP 2-D embedding
    embedding = get_embedding(data)
    # draw_umap(embedding, true_labels)
    
    # K-means for clustering umi embedings
    kmeans = KMeans(n_clusters=n_clusters).fit(embedding)
    labels = kmeans.labels_

    # Calculate metrics
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)
    sc = silhouette_score(embedding, labels)

    print(f'Adjusted Rand Index: {ari}')
    print(f'Normalized Mutual Information: {nmi}')
    print(f'Silhouette Coefficient: {sc}')
    
    return ari, nmi, sc, embedding


def draw_umap(embedding, label, figname=None, output_dir=None, figsize=(15, 10), labelsize=12):  

    palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(label)))
    fig, ax = plt.subplots(figsize=figsize)

    plt.xlim([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 1.5])
    plt.ylim([np.min(embedding[:, 1]) - 0.5, np.max(embedding[:, 1]) + 0.5])
    plt.xlabel('UMAP 1', fontsize=labelsize)
    plt.ylabel('UMAP 2', fontsize=labelsize)
    plt.title(figname)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=label, 
                hue_order=list(np.unique(label)), palette=palette)
    leg = plt.legend(prop={'size': labelsize}, loc='upper right', markerscale=2.00)
    leg.get_frame().set_alpha(0.9)
    plt.setp(ax, xticks=[], yticks=[])
    plt.show()
    if output_dir is not None:
        import os
        fig.savefig(os.path.join(output_dir, figname+ '.png'))

    
def draw_multiple_umap(embeddings, label, names=None, output_dir=None, denoise_method=None, labelsize=14):
    
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
    names = [names for i in range(len(embeddings))] if names is None else names
    
    palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(label)))
    fig, ax = plt.subplots(1, len(embeddings), figsize=(len(embeddings)*12.5, 10))

    for (i, embed), figname in zip(enumerate(embeddings), names):
        ax[i].set_xlim([np.min(embed[:, 0]) - 0.5, np.max(embed[:, 0]) + 1.5])
        ax[i].set_ylim([np.min(embed[:, 1]) - 0.5, np.max(embed[:, 1]) + 0.5])
        ax[i].set_xlabel('UMAP 1', fontsize=labelsize)
        ax[i].set_ylabel('UMAP 2', fontsize=labelsize)
        ax[i].set_title(figname)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        
        sns.scatterplot(x=embed[:, 0], y=embed[:, 1], hue=label, 
                    hue_order=list(np.unique(label)), palette=palette, ax=ax[i])
        leg = ax[i].legend(prop={'size': labelsize}, loc='upper right', markerscale=2.00)
        leg.get_frame().set_alpha(0.9)
        plt.setp(ax[i], xticks=[], yticks=[])
    plt.show()
    if output_dir is not None:
        import os
        fig.savefig(os.path.join(output_dir, denoise_method+'-results.png'))

class Clustering():
    def __init__(self, adata, denoise_method, 
                label, random_seed=None, mode=['raw','denoise'],
                show_plot=False):
        self.adata = adata
        self.denoise_method = denoise_method
        self.label = label
        self.mode = mode
        self.random_seed = random_seed
        self.show_plot = show_plot
        self.n_clusters = len(np.unique(self.label))
        if not isinstance(self.random_seed, list):
            self.random_seed = [self.random_seed]
        
        self.datasets = []
        if 'raw' in self.mode:
            self.datasets.append(adata.raw.X) # raw data
        if 'denoise' in self.mode:
            self.datasets.append(adata.X) # denoised data
        if 'latent' in self.mode:
            self.datasets.append(adata.obsm['X_hidden']) # latent repre
        
        result = []
        self.embeddings = []
        self.fignames = []
        for name, data in zip(self.mode, self.datasets):
            for seed in self.random_seed:
            
                print("seed: %d  Evaluating clustering results for %s data..." % (seed, name))
                
                ari, nmi, sc, embedding = calculate_cluster_results(data, self.label, seed)
                result.append([self.denoise_method, name, seed, ari, nmi, sc])
                figname = '-'.join(str(s).lower() for s in [self.denoise_method, name, seed])
                self.fignames.append(figname)
                self.embeddings.append(embedding)
                
        self.output = np.array(result)
        self.output = pd.DataFrame(self.output, columns=["denoise method", "dataname", "random seed", "ARI", "NMI", "SC"])
        
    def get_output(self):
        return self.output
    
    def get_umap(self,
                method=None, mode=None, seed=None, same_figure=False, output_dir=None):
        search = lambda string, key: True if string is None else any([str(s) in key for s in string])
        
        inds = [
            (search(method, name) and search(mode, name) and search(seed, name)) for name in self.fignames]
        from itertools import compress
        
        embeddings = list(compress(self.embeddings, inds))
        fignames = list(compress(self.fignames,inds))
        label = self.label
        
        if same_figure:
            draw_multiple_umap(embeddings, label, fignames, output_dir)
        else:
            for figname, emb in zip(fignames, embeddings):
                draw_umap(emb, label, figname, output_dir, self.denoise_method)


# def draw_umap(embedding, label):
#     contour_c = '#444444'
#     labelsize = 25
#     label = label
#     palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(label)))

#     fig, ax = plt.subplots(figsize=(20, 15))

#     plt.xlim([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 1.5])
#     plt.ylim([np.min(embedding[:, 1]) - 0.5, np.max(embedding[:, 1]) + 0.5])
#     plt.xlabel('UMAP 1', fontsize=labelsize)
#     plt.ylabel('UMAP 2', fontsize=labelsize)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
    
#     # plt.scatter(embedding[:, 0], embedding[:, 1], lw=0, , label=label, alpha=1.0, s=180, linewidth=2)
#     sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=label, 
#                 hue_order=list(np.unique(label)), palette=palette)
#     leg = plt.legend(prop={'size': labelsize}, loc='upper right', markerscale=2.00)
#     leg.get_frame().set_alpha(0.9)
#     plt.setp(ax, xticks=[], yticks=[])
#     plt.show()
    
    
# def draw_multiple_umap(embedding, label, ax=None, figsize=(15, 10), labelsize=12):
#     plt.close()
#     palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(label)))
#     if isinstance(embedding, list):
#         fig, ax = plt.subplots(1, len(embedding), figsize=(len(embedding)*12.5, 10))

#     for i, embed in enumerate(embedding):
#         ax[i].set_xlim([np.min(embed[:, 0]) - 0.5, np.max(embed[:, 0]) + 1.5])
#         ax[i].set_ylim([np.min(embed[:, 1]) - 0.5, np.max(embed[:, 1]) + 0.5])
#         ax[i].set_xlabel('UMAP 1', fontsize=labelsize)
#         ax[i].set_ylabel('UMAP 2', fontsize=labelsize)

#         ax[i].spines['right'].set_visible(False)
#         ax[i].spines['top'].set_visible(False)
        
#         sns.scatterplot(x=embed[:, 0], y=embed[:, 1], hue=label, 
#                     hue_order=list(np.unique(label)), palette=palette, ax=ax[i])
#         leg = ax[i].legend(prop={'size': labelsize}, loc='upper right', markerscale=2.00)
#         leg.get_frame().set_alpha(0.9)
#         plt.setp(ax[i], xticks=[], yticks=[])
#     plt.show()    


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

