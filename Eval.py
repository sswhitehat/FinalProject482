import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import csv
from sklearn.metrics import accuracy_score

from LSTM import HybridBoxingLSTM, KeypointDataset, parse_annotations, STRIKE_TYPES, STRIKE_TYPE_TO_ID

def load_model(model_path, input_size, hidden_size, num_layers, device):
    model = HybridBoxingLSTM(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_data_and_predict(model_path, csv_file_path, annotations, input_size, hidden_size, num_layers, device):
    # Load CSV file into a DataFrame
    data_frame = pd.read_csv(csv_file_path)
    # Create dataset from DataFrame
    dataset = KeypointDataset(data_frame, annotations)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    model = load_model(model_path, input_size, hidden_size, num_layers, device)
    predictions, actuals, frames = predict(model, loader, device)
    return predictions, actuals, frames

def predict(model, loader, device):
    predictions = []
    actual_strikes = []
    frame_numbers = []
    with torch.no_grad():
        for data in loader:
            keypoints = data['keypoints'].to(device)
            labels = data['labels'].cpu().numpy()
            frames = data['Frame Number']
            outputs = model(keypoints)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            actual_strikes.extend(labels)
            frame_numbers.extend(frames)
    return predictions, actual_strikes, frame_numbers

def evaluate_accuracy_per_strike_type(predictions, actual_strikes):
    accuracies = {}
    for strike_id, strike_name in enumerate(STRIKE_TYPES):
        actual_array = (np.array(actual_strikes) == strike_id)
        pred_array = (np.array(predictions) == strike_id)
        if actual_array.any():
            accuracy = accuracy_score(actual_array, pred_array)
            accuracies[strike_name] = accuracy
    return accuracies

def save_predictions_to_csv(predictions, actual_strikes, frames, csv_output_path):
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Number', 'Predicted Strike', 'Actual Strike'])
        for frame, pred, actual in zip(frames, predictions, actual_strikes):
            writer.writerow([frame, STRIKE_TYPES[pred], STRIKE_TYPES[actual]])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Model/model_fold_5_epoch_50.pth'
    csv_file = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Keypoint Files/SuperbonvPetrosyanKeypoints.csv'
    annotations = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Annotations/SuperbonvPetrysan.xml'
    csv_output_path = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Model Output/Superbon_output_predictions.csv'

    # Load annotations from XML
    annotations_dict = parse_annotations(annotations)

    # Load data and predict
    predictions, actual_strikes, frames = load_data_and_predict(
        model_path, csv_file, annotations_dict, 51, 128, 2, device
    )

    # Save predictions to CSV
    save_predictions_to_csv(predictions, actual_strikes, frames, csv_output_path)

    # Evaluate and print accuracy per strike type
    accuracies = evaluate_accuracy_per_strike_type(predictions, actual_strikes)
    print("Accuracy per strike type:", accuracies)
