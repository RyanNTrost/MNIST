import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm


# Problem Information
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_INPUT = 28 * 28
N_CLASSES = 10

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 256
N_HIDDEN = 10
EPOCHS = 10


def train(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f'Epoch: {epoch}/{EPOCHS}'):
            images = images.reshape(images.shape[0], -1)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


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
        'train': DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True),
        'test': DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    }

    return loaders


def main():
    # Get the data
    loaders = get_data_loaders()

    # Create the model
    model = NN(N_INPUT, N_HIDDEN, N_CLASSES).to(DEVICE)

    # Train the model on the data
    train(model, loaders['train'])

    train_accuracy = compute_accuracy(model, loaders['train'])
    test_accuracy = compute_accuracy(model, loaders['test'])

    print(f'Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')


if __name__ == '__main__':
    main()
