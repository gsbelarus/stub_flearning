from typing import Union, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BCELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from client.dataset import load_dataset, preprocess_data, TransactionsDataset, preprocess_set
from client.net import Net
from joint_ml import Metric

model_parameters = {
    'n_features': 30,
    'hidden_dim': 32
}

dataset_parameters = {
    'shuffle': True,
    'test_size': 0.1
}

train_parameters = {
    'epochs': 15,
    'batch_size': 16,
    'lr': 0.0001
}

test_parameters = {

}


def load_model(n_features=30, hidden_dim=32) -> nn.Module:
    model = Net(n_features, hidden_dim)
    return model


def get_dataset(dataset_path: str, with_split: bool, test_size: float, shuffle: bool) -> Union[
    Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset],
    Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset], Tuple[torch.utils.data.Dataset]]:
    transactions, labels = load_dataset(dataset_path)
    if with_split:
        x_train, x_test, y_train, y_test = train_test_split(transactions, labels, test_size=test_size, shuffle=shuffle)
        x_train, x_test = preprocess_data(x_train, x_test)

        train_set = TransactionsDataset(x_train, y_train)
        test_set = TransactionsDataset(x_test, y_test)

        return train_set, test_set
    else:
        x_test = preprocess_set(transactions)
        test_set = TransactionsDataset(x_test, labels)
        return test_set


def train(model: torch.nn.Module, train_set: torch.utils.data.Dataset, epochs: int, batch_size: int, lr: float) -> \
Tuple[List[Metric], torch.nn.Module]:
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(params=model.parameters(), lr=lr)
    loss_fn = BCELoss()

    train_epoch_loss_metric = Metric(name="train_epoch_loss")

    model.train()

    for epoch in range(epochs):
        train_epoch_loss = 0.0
        model.train()

        for i, data in enumerate(train_dataloader):
            transactions, labels = data['transaction'], data['label']
            transactions = transactions.reshape(transactions.shape[0], 1, transactions.shape[1])

            optimizer.zero_grad()

            output = model(transactions)

            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        train_epoch_loss /= len(train_dataloader)
        train_epoch_loss_metric.log_value(train_epoch_loss)

    return ([train_epoch_loss_metric], model)


def test(model: torch.nn.Module, test_set: torch.utils.data.Dataset, return_output: bool) -> Union[
    List[Metric], Tuple[List[Metric], list]]:
    test_loss = 0.0
    model.eval()
    loss_fn = BCELoss()

    test_dataloader = DataLoader(test_set)

    outputs = []
    labels = np.array([])

    for i, data in enumerate(test_dataloader):
        transactions, label = data['transaction'], data['label']

        transactions = transactions.reshape(transactions.shape[0], 1, transactions.shape[1])
        output = model(transactions)

        loss = loss_fn(output, label)

        test_loss += loss.item()

        outputs.append(output.cpu().detach().numpy().reshape(-1))
        labels = np.hstack([labels, label.cpu().reshape(-1)])

    test_loss /= len(test_dataloader)

    test_loss_metric = Metric(name="test_loss")
    test_loss_metric.log_value(test_loss)

    if return_output:
        return [test_loss_metric], outputs
    else:
        return [test_loss_metric]

def get_prediction(model: torch.nn.Module, dataset_path: str) -> list:
    test_set = get_dataset(dataset_path, with_split=False, test_size=1, shuffle=False)

    metrics, output = test(model, test_set, return_output=True)

    return output
