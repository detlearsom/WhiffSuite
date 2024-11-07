import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import deque
import numpy as np
import random

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        is_same_class = index % 2 == 1
        if is_same_class:
            label = 1.0
            class_idx = random.randint(0, 1)
            index1, index2 = random.choices(np.where(self.y_data == class_idx)[0], k=2)
        else:
            label = 0.0
            index1 = random.choice(np.where(self.y_data == 0)[0])
            index2 = random.choice(np.where(self.y_data == 1)[0])

        return (self.X_data[index1], self.X_data[index2], torch.tensor([label], dtype=torch.float32))

    def __len__(self):
        return len(self.X_data)


class SimpleSiamese(nn.Module):
    def __init__(self, input_shape):
        super(SimpleSiamese, self).__init__()
        self.simple = nn.Sequential(
            nn.Linear(input_shape, 196),
            nn.ReLU(),
            nn.Linear(196, 196),
            nn.ReLU(),
            nn.Linear(196, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.sig = nn.Sequential(
            nn.Linear(64, 16),
            nn.Sigmoid()
        )

        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        x = self.simple(x)
        return self.sig(x.view(x.size(0), -1))

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        distance = torch.abs(output1 - output2)
        return self.sigmoid(self.out(distance))


def train_siamese_network(train_features, train_labels, test_features, test_labels, epochs=5):
    x_tensor_train = torch.tensor(train_features.values).float()
    y_tensor_train = torch.tensor(train_labels.values).long()
    x_tensor_test = torch.tensor(test_features.values).float()
    y_tensor_test = torch.tensor(test_labels.values).long()

    device = torch.device("cpu")
    print(f'Device: {device}')

    train_dataset = ClassifierDataset(x_tensor_train, y_tensor_train)
    test_dataset = ClassifierDataset(x_tensor_test, y_tensor_test)


    input_shape = x_tensor_train.size(1)
    model = SimpleSiamese(input_shape).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=5)
    test_loader = DataLoader(test_dataset, batch_size=5)
    queue = deque(maxlen=epochs)
    epoch_accuracies = []
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        epoch_loss = 0.0
        for batch_id, (img1, img2, label) in enumerate(train_loader, 1):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


            if batch_id % 10 == 0:
                correct, total = 0, 0
                model.eval()
                with torch.no_grad():
                    for test1, test2, label in test_loader:
                        test1, test2 = test1.to(device), test2.to(device)
                        pred = model(test1, test2).cpu().numpy()
                        predicted_label = np.heaviside(pred - 0.5, 0)
                        correct += np.sum(predicted_label == label.numpy())
                        total += len(label)


                epoch_accuracy = correct / total
                epoch_accuracies.append(epoch_accuracy)
                print(f'Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f} - Test Accuracy: {epoch_accuracy:.4f}')

        final_accuracy = np.mean(epoch_accuracies)
        print("#" * 70)
        print(f'Average Accuracy Over Epochs: {final_accuracy:.4f}')

    return epoch_accuracies, final_accuracy