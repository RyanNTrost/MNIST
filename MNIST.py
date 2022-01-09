import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
import os
from draw import ImageGenerator
import tkinter as tk

# Problem Information
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_INPUT = 28 * 28
N_CLASSES = 10
EPOCHS = 10
BATCH_SIZE = 1
N_TRIALS = 100
NUM_WORKERS = 8
MODEL_SAVE_PATH = 'models'
LOAD_MODEL = True
MODEL_TO_LOAD = 'best_fcn.pth'


def define_model(params):
    return NN(N_INPUT, params['n_hidden'], N_CLASSES)


def objective(trial):
    # Setup Hyperparameters
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-1),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop']),
        'n_hidden': trial.suggest_int('n_hidden', 1, 10000)
    }

    # Get data
    loaders = get_data_loaders()

    # Define model
    model = define_model(params).to(DEVICE)

    # Train the model
    accuracy = train(trial, params, model, loaders['train'])

    save_model(model, trial.number)

    return accuracy


def train(trial, params, model, train_loader):
    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        num_correct = 0
        total = 0

        for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False,
                                           desc=f'Epoch: {epoch}/{EPOCHS}'):
            images = images.reshape(images.shape[0], -1)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Compute Accuracy
            _, indices = outputs.max(1)
            num_correct += (labels == indices).sum()
            total += indices.size(0)

        accuracy = (num_correct / total).item()
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def compute_accuracy(model, loader):
    model.eval()

    num_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Computing Accuracy'):
            images = images.reshape(images.shape[0], -1)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, indices = outputs.max(1)
            num_correct += (labels == indices).sum()
            total += indices.size(0)

    model.train()

    return (num_correct / total).item()


class NN(nn.Module):
    def __init__(self, n_input, n_hidden, n_classes):
        super(NN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_data_loaders():
    train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

    loaders = {
        'train': DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True),
        'test': DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    }

    return loaders


def save_model(model, trial_number):
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    torch.save(model, f'{MODEL_SAVE_PATH}/model_{trial_number}.pth')


def load_model(trial_number):
    return torch.load(f'{MODEL_SAVE_PATH}/{MODEL_TO_LOAD}')


def main():

    if LOAD_MODEL:
        model = load_model(MODEL_TO_LOAD)
        loaders = get_data_loaders()

        root = tk.Tk()
        root.wm_geometry("%dx%d+%d+%d" % (400, 400, 10, 10))
        root.config(bg='white')
        ImageGenerator(root, 10, 10, model)
        root.mainloop()

        # train_accuracy = compute_accuracy(model, loaders['train'])
        # test_accuracy = compute_accuracy(model, loaders['test'])
        #
        # print('Train Accuracy:', train_accuracy)
        # print('Test Accuracy:', test_accuracy)

    else:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)


if __name__ == '__main__':
    main()
