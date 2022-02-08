from pathlib import Path
import pandas as pd
import numpy as np

from gcn_package.features.graph_construction import make_group_graph
from gcn_package.data.time_windows_dataset import TimeWindowsDataset
from gcn_package.models.gcn import HaoGCN

# some parameters here
dimension = 1024
window_length = 20
random_seed = 0
batch_size = 16
epochs = 30

dic_labels = {'adult': 0, 'child': 1}

data_path = Path(__file__).parents[1] / "data/processed/"
# get subject id and label
participants = pd.read_csv(data_path / "participants.tsv", index_col=0, header=0, sep='\t')
labels = participants['Child_Adult']

# select balanced child and adult
participants = participants[89: ]
labels = participants['Child_Adult']

# make a group graph
connectomes = []
for subject in labels.index:
    path = data_path / "dataset-ds000228_timeseries" / subject / \
        f"{subject}_task-pixar_atlas-DiFuMo{dimension}dimensionsSegmented_desc-deconfounds_connectome.tsv"
    conn = pd.read_csv(path, index_col=0, header=0, sep='\t').values
    connectomes.append(conn.astype(np.float32))

graph = make_group_graph(connectomes, k=8)

# split the data by time window size and save to file

label_df = pd.DataFrame(columns=['label', 'filename'])
split_twindow_dir = data_path / 'split_timewindow'
split_twindow_dir.mkdir(parents=True, exist_ok=True)

for subject in labels.index:
    label = labels[subject]
    ts_path = data_path / "dataset-ds000228_timeseries" / subject / \
        f"{subject}_task-pixar_atlas-DiFuMo{dimension}dimensionsSegmented_desc-deconfounds_timeseries.tsv"
    ts_data = pd.read_csv(ts_path, index_col=False, header=0, sep='\t').values.astype(np.float32)
    ts_duration = ts_data.shape[0]
    ts_filename = f"{subject}_{label}_seg"
    valid_label = dic_labels[label]

    # Split the timeseries
    rem = ts_duration % window_length
    n_splits = int(np.floor(ts_duration / window_length))

    ts_data = ts_data[:(ts_duration - rem), :]

    for j, split_ts in enumerate(np.split(ts_data, n_splits)):
        ts_output_file_name = split_twindow_dir / '{}_{:04d}.npy'.format(ts_filename, j)

        split_ts = np.swapaxes(split_ts, 0, 1)
        np.save(ts_output_file_name, split_ts)

        curr_label = {'label': valid_label, 'filename': ts_output_file_name.name}
        label_df = pd.concat((label_df, pd.DataFrame([curr_label])))

label_df.to_csv(split_twindow_dir / 'labels.csv', index=False)


train_dataset = TimeWindowsDataset(
    data_dir=split_twindow_dir,
    partition="train",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

valid_dataset = TimeWindowsDataset(
    data_dir=split_twindow_dir,
    partition="valid",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

test_dataset = TimeWindowsDataset(
    data_dir=split_twindow_dir,
    partition="test",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

print("train dataset: {}".format(train_dataset))
print("valid dataset: {}".format(valid_dataset))
print("test dataset: {}".format(test_dataset))

import torch
from torch.utils.data import DataLoader


torch.manual_seed(random_seed)
train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_generator = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


gcn = HaoGCN(graph.edge_index,
            graph.edge_attr,
            batch_size=batch_size,
            n_roi=conn.shape[0],
            n_timepoints=window_length,
            n_classes=2)

# NOTE - Early stopping: https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.float()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * dataloader.batch_size

        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= X.shape[0]
        if (batch % 10 == 0) or (current == size):
            print(f"#{batch:>5};\ttrain_loss: {loss:>0.3f};\ttrain_accuracy:{(100*correct):>5.1f}%\t\t[{current:>5d}/{size:>5d}]")


def valid_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size
    correct /= size

    return loss, correct


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-4, weight_decay=5e-4)

for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    train_loop(train_generator, gcn, loss_fn, optimizer)
    loss, correct = valid_test_loop(valid_generator, gcn, loss_fn)
    print(f"Valid metrics:\n\t avg_loss: {loss:>8f};\t avg_accuracy: {(100*correct):>0.1f}%")

# results
loss, correct = valid_test_loop(test_generator, gcn, loss_fn)
print(f"Test metrics:\n\t avg_loss: {loss:>f};\t avg_accuracy: {(100*correct):>0.1f}%")
