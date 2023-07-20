import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.utils as utils
from dataset import HW3Dataset
import matplotlib.pyplot as plt
from utils import create_subgraph, create_mask, compute_class_weight



#model parameters
NUM_FEATURES = 128
NUM_CLASSES = 40
HIDDEN_DIM = 64

#training parameters
LEARNING_RATE = 0.005
NUM_EPOCHS = 350
DECAY = 5e-4
HEADS=12

# Définition du modèle GNN
class GATClassifierLinear(nn.Module):
    def __init__(self, hidden_dim, heads=HEADS):
        super(GATClassifierLinear, self).__init__()
        self.gat1 = GATv2Conv(NUM_FEATURES, hidden_dim, heads=heads)
        self.gat2 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=heads)
        self.linear1 = nn.Linear(hidden_dim*heads, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, NUM_CLASSES)


    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h = self.linear1(h)
        h = F.elu(h)
        h = self.linear2(h)
        return F.log_softmax(h, dim=1)


class GATClassifierClassic(nn.Module):
    def __init__(self, hidden_dim, heads=HEADS):
        super(GATClassifierClassic, self).__init__()
        self.gat1 = GATv2Conv(NUM_FEATURES, hidden_dim, heads=heads)
        self.gat2 = GATv2Conv(hidden_dim*heads, NUM_CLASSES, 1)


    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)


def train(model, criterion, optimizer, data, mask_train):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask_train], data.y[mask_train].squeeze())
    loss.backward()
    optimizer.step()
    return loss

def test(model, data, mask_test=None):
    model.eval()
    total_correct = 0
    out = model(data.x, data.edge_index)
    prediction = out.argmax(dim=1)
    if mask_test !=None:
        test_prediction = prediction[mask_test]
        real_label = data.y[mask_test]
    else:
        test_prediction = prediction
        real_label = data.y
    for i in range(0, len(test_prediction)):
        y_pred = test_prediction[i].item()
        y_real = real_label[i].squeeze().item()
        if y_pred == y_real:
            total_correct+=1
    test_accuracy = total_correct/len(real_label)
    return test_accuracy, test_prediction

if __name__ == '__main__':

    # Préparation des données
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    #class_weight = compute_class_weight(data.y, NUM_CLASSES)
    hidden_dim = 64

    model = GATClassifierLinear(HIDDEN_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    criterion = nn.CrossEntropyLoss()

    mask_train, mask_test = create_mask(data)
    losses = []
    accuracies = []

    for epoch in range(0, NUM_EPOCHS):
        loss = train(model, criterion, optimizer, data, mask_train)
        losses.append(loss.item())
        print(f'Epoch: {epoch}, loss: {loss: .4f}')
        if epoch%10==0:
            accuracy, prediction = test(model, data, mask_test)
            accuracies.append(accuracy)
            print(f"Accuracy: {accuracy}")
            torch.save(model.state_dict(), "model_epoch_"+str(epoch)+".pt")


    axis_x = [i for i in range(0, NUM_EPOCHS, 1)]
    plt.plot(axis_x, losses, color='red')
    plt.title("loss evolution by epochs")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    axis_x = [i for i in range(0, int(NUM_EPOCHS/10))]
    plt.plot(axis_x, accuracies)
    plt.title("accuracy evolution by epochs", color='blue')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()


