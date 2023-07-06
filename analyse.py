import networkx as nx
import matplotlib.pyplot as plt
from dataset import HW3Dataset
import torch

def analyze_graph(edge_index):
    num_nodes = edge_index.max() + 1
    degree = torch.zeros(num_nodes)

    for edge in edge_index.t():
        src, dst = edge
        degree[src] += 1
        degree[dst] += 1

    avg_connections = degree.mean().item()

    top_10_nodes = degree.argsort(descending=True)[:10]
    top_10_degrees = degree[top_10_nodes].tolist()

    return avg_connections, top_10_nodes, top_10_degrees



if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]


    classes = data.y.squeeze().tolist()
    class_counts = {}
    for c in classes:
        class_counts[c] = class_counts.get(c, 0) + 1

    plt.figure(figsize=(8, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel("Classes")
    plt.ylabel("Number of nodes")
    plt.title("Class repartiton")
    plt.show()

    avg_connections, top_10_nodes, top_10_degrees = analyze_graph(data.edge_index)
    print("mean number of edges per nodes:", avg_connections)
    print("Top 10 of the most linked nodes :")
    for node, degree in zip(top_10_nodes, top_10_degrees):
        print("Node :", node.item(), " - Degree :", degree)
