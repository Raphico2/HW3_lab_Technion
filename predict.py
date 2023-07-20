import torch
from dataset import HW3Dataset
from GAT_classification import GATClassifierLinear, test
import pandas as pd
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
    model.load_state_dict(torch.load('model_epoch_340.pt'))
    accuracy, pred = test(model, data)
    predictions = {'idx': [], 'prediction': []}
    for i in range(len(pred)):
        predictions['idx'].append(i)
        predictions['prediction'].append(pred[i].item())
    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv('prediction.csv', index=False)
    print("accuracy : " + str(accuracy))


