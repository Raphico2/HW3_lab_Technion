import torch
from dataset import HW3Dataset
from GAT_classification import GATClassifierLinear, test
from utils import create_mask

#model parameters
NUM_FEATURES = 128
NUM_CLASSES = 40
HIDDEN_DIM = 64
HEADS=12


if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = GATClassifierLinear(HIDDEN_DIM, heads=HEADS).to(device)
    model.load_state_dict(torch.load('model_epoch_330.pt'))
    mask_train, mask_test = create_mask(data)
    accuracy = test(model, data, mask_test)
    print("accuracy : " + str(accuracy))


