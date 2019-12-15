import torch
import pandas as pd

from model import TinyClassifier2d
from load_data import load_data

def model_test(model, loader, device):
    df = pd.DataFrame(columns=['id', 'label'])

    model.to(device)
    with torch.no_grad():
        for data in loader:
            # get inputs and labels from data
            inputs, _, img = data
            inputs = inputs.to(device)

            # model output
            outputs = model(inputs)
            predicted = torch.max(outputs.data,1)[1]

            # convert to list
            predicted = predicted.tolist()
            img = list(img)

            for id, label in zip(img, predicted):
                df = df.append({'id': id, 'label': label}, ignore_index=True)
    return df


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Using device:', device)

_, _, test_loader = load_data()

best_epoch = 30
best = torch.load('./save/tiny/train/epoch-{:02d}.pth'.format(best_epoch), map_location=device)
tiny = TinyClassifier2d()
tiny.load_state_dict(best)

df = model_test(tiny, test_loader, device)
df.to_csv('tiny_result.csv', index=False)
