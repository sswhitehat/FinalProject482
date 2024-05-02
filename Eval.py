import torch
import pandas as pd
from torch.utils.data import DataLoader
import csv

from LSTM import HybridBoxingLSTM, KeypointDataset, parse_annotations, STRIKE_TYPES


# Assuming the existence of `HybridBoxingLSTM`, `KeypointDataset`, `STRIKE_TYPES`

def load_model(model_path, input_size, hidden_size, num_layers, device):
    model = HybridBoxingLSTM(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_data_and_predict(model_path, csv_file, annotations, input_size, hidden_size, num_layers, device):
    dataset = KeypointDataset(csv_file, annotations)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    model = load_model(model_path, input_size, hidden_size, num_layers, device)
    predictions, actuals = predict(model, loader, device)
    return predictions, actuals

def predict(model, loader, device):
    predictions = []
    actual_strikes = []
    with torch.no_grad():
        for data in loader:
            keypoints = data['keypoints'].to(device)
            actual_strikes.extend(data['labels'].cpu().numpy())  # Assuming labels are available in the dataset
            outputs = model(keypoints)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions, actual_strikes

def save_predictions_to_csv(predictions, actual_strikes, csv_output_path):
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Number', 'Predicted Strike', 'Actual Strike'])
        for i, (pred, actual) in enumerate(zip(predictions, actual_strikes)):
            writer.writerow([i, STRIKE_TYPES[pred], STRIKE_TYPES[actual]])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Model/model_epoch_0.pth'
    csv_file = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Keypoint Files/SuperbonvPetrosyanKeypoints.csv'
    annotations = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Annotations/SuperbonvPetrysan.xml'  # This should be loaded appropriately
    csv_output_path = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Model Output/output_predictions.csv'

    # Assuming the functions to parse XML and handle data are already defined
    annotations_dict = parse_annotations(annotations)
    input_size = 34  # Example input size
    hidden_size = 128  # Example hidden size
    num_layers = 2  # Example number of LSTM layers

    predictions, actual_strikes = load_data_and_predict(model_path, csv_file, annotations_dict, input_size, hidden_size, num_layers, device)
    save_predictions_to_csv(predictions, actual_strikes, csv_output_path)

