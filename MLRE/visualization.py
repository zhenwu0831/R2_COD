import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os


def plot_pca(text_embs, node_embs, epoch, args):
    pca = PCA(n_components=2)
    all_embs = torch.cat([text_embs, node_embs], dim=0).cpu().detach().numpy()
    pca_result = pca.fit_transform(all_embs)
    
    text_pca = pca_result[:len(text_embs)]
    node_pca = pca_result[len(text_embs):]

    plt.figure(figsize=(8, 6))
    plt.scatter(text_pca[:, 0], text_pca[:, 1], label='Text Embs', alpha=0.6)
    plt.scatter(node_pca[:, 0], node_pca[:, 1], label='Graph Embs', alpha=0.6)
    plt.legend()
    plt.title(f'PCA of Text and Graph Embeddings at Epoch {epoch}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)


    plt.savefig(args.img_path)
    plt.close()
    plt.clf()

def calculate_cosine_similarity(projected1, projected2):
    # Ensure embeddings are normalized for cosine similarity
    projected1 = F.normalize(projected1, p=2, dim=1)
    projected2 = F.normalize(projected2, p=2, dim=1)
    
    # Determine the minimum batch size
    min_size = min(projected1.size(0), projected2.size(0))
    
    # Truncate both tensors to match the smaller batch size
    projected1 = projected1[:min_size]
    projected2 = projected2[:min_size]
        
    # Compute cosine similarity (element-wise)
    cosine_sim = F.cosine_similarity(projected1, projected2, dim=1)
    return cosine_sim


def compute_within_between_distances(embs1, embs2, metric="cosine"):


    embs1 = embs1.cpu().detach().numpy()
    embs2 = embs2.cpu().detach().numpy()

    # Compute within-group distances
    within_group1 = cdist(embs1, embs1, metric).mean()
    within_group2 = cdist(embs2, embs2, metric).mean()

    # Compute between-group distances
    between_groups = cdist(embs1, embs2, metric).mean()

    return {
        "within_group1": within_group1,
        "within_group2": within_group2,
        "between_groups": between_groups,
    }
