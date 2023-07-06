import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.utils as utils
from dataset import HW3Dataset
from utils import create_subgraph, create_mask, compute_class_weight
import matplotlib.pyplot as plt


#model parameters
NUM_FEATURES = 128
NUM_CLASSES = 40
HIDDEN_DIM = 64

#training parameters
LEARNING_RATE = 0.01
NUM_EPOCHS = 700
DECAY = 5e-4
BATCH_SIZE = 64

# Définition du modèle GNN
class GNNClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(NUM_FEATURES, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


def train(model, criterion, optimizer, data, mask_train):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask_train], data.y[mask_train].squeeze())
    loss.backward()
    optimizer.step()
    return loss

def test(model, data, mask_test):
    model.eval()
    total_correct = 0
    out = model(data.x, data.edge_index)
    prediction = out.argmax(dim=1)
    test_prediction = prediction[mask_test]
    real_label = data.y[mask_test]
    for i in range(0, len(test_prediction)):
        y_pred = test_prediction[i].item()
        y_real = real_label[i].squeeze().item()
        if y_pred == y_real:
            total_correct+=1
    test_accuracy = total_correct/len(real_label)
    return test_accuracy


if __name__ == '__main__':

    # Préparation des données
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    #class_weight = compute_class_weight(data.y, NUM_CLASSES)
    hidden_dim = 64

    model = GNNClassifier(HIDDEN_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    criterion = nn.CrossEntropyLoss()

    mask_train, mask_test = create_mask(data)
    losses = []
    accuracies = []

    for epoch in range(0, NUM_EPOCHS):
        loss = train(model, criterion, optimizer, data, mask_train)
        losses.append(loss)
        print(f'Epoch: {epoch}, loss: {loss: .4f}')
        if epoch%10==0:
            accuracy = test(model, data, mask_test)
            accuracies.append(accuracy)
            print(f"Accuracy: {accuracy}")

    axis_x = [i for i in range(0, NUM_EPOCHS)]
    plt.plot(axis_x, losses, color='red')
    plt.title("loss evolution by epochs")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    axis_x = [i for i in range(0, NUM_EPOCHS, 10)]
    plt.plot(axis_x, accuracies)
    plt.title("accuracy evolution by epochs", color='blue')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()
