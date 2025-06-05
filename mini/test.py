import torch
from model import AODModel
from aod_dataset import AODDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def regression_accuracy(y_true, y_pred, tolerance=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.abs(y_true - y_pred) <= tolerance
    return np.mean(correct) * 100  # accuracy in %

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc=96.98712
    model = AODModel().to(device)
    model.load_state_dict(torch.load('models/aod_model.pth'))
    model.eval()

    train_dataset = AODDataset(csv_path='data/labels.csv', image_dir='data/train/')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            y_true.append(label.item())
            y_pred.append(output.item())

    # Convert to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
      # Adjust tolerance as needed

    # Output
    print("\n=== Evaluation ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Accuracy : {acc:.4f}%")

if __name__ == "__main__":
    test()
