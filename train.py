import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from aod_dataset import AODDataset
from model import AODModel
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = AODDataset(csv_path='data/labels.csv', image_dir='data/train')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = AODModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 110
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    

if __name__ == "__main__":
    train()
