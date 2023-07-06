import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.utils as utils
import networkx as nx
import matplotlib.pyplot as plt
from dataset import HW3Dataset


def create_subgraph(edge_index, selected_nodes):
    """
    The function create a subgraph containing only some selected nodes
    :param edges:
    :param selected_nodes:
    :return: the subgraph
    """
    mask = torch.isin(edge_index[0], selected_nodes) & torch.isin(edge_index[1], selected_nodes)
    sub_edge_index = edge_index[:, mask]
    return sub_edge_index

def create_mask(data):
    """
    Create real mask for separate train and test set
    :param data:
    :return: the train and test mask
    """
    mask_train = torch.zeros(100000, dtype=torch.bool)
    mask_test = torch.zeros(100000, dtype=torch.bool)
    mask_train[data.train_mask] = 1
    mask_test[data.val_mask] = 1
    return mask_train, mask_test

def compute_class_weight(labels, nb_classes):
    class_counts = torch.zeros(nb_classes)
    for label in labels:
        class_counts[label] += 1
    for i in range(len(class_counts)):
        class_counts[i] = 1/(class_counts[i])

    class_weight = class_counts/torch.sum(class_counts)
    return class_weight


if __name__ == '__main__':

    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]

    selected_nodes = data.train_mask[:800]
    print(selected_nodes)
    sub_edge_index = create_subgraph(data.edge_index, selected_nodes)
    print(sub_edge_index.size(0))
    print(sub_edge_index.size(1))
    graph = nx.DiGraph()
    graph.add_edges_from(sub_edge_index.t().tolist())

    # Visualiser le graphe
    plt.figure(figsize=(10, 8))
    nx.draw_networkx(graph, with_labels=False, node_size=10)
    plt.title("Graph visualisation")
    plt.show()
