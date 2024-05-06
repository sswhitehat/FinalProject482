import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import csv
from sklearn.metrics import accuracy_score

from LSTM import KickBoxingLSTM, KeypointDataset, parse_annotations, STRIKE_TYPES, STRIKE_TYPE_TO_ID

def load_model(model_path, input_size, hidden_size, num_layers, device):
    # Initialize the LSTM model with specified architecture parameters
    model = KickBoxingLSTM(input_size, hidden_size, num_layers)
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Move model to the specified device (GPU or CPU)
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    return model

def load_data_and_predict(model_path, csv_file_path, annotations, input_size, hidden_size, num_layers, device):
    # Load CSV file into a DataFrame
    data_frame = pd.read_csv(csv_file_path)
    # Create dataset from DataFrame and annotations
    dataset = KeypointDataset(data_frame, annotations)
    # DataLoader to manage batches of data
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    # Load the model with the specified architecture and device
    model = load_model(model_path, input_size, hidden_size, num_layers, device)
    # Perform predictions using the model
    predictions, actuals, frames = predict(model, loader, device)
    return predictions, actuals, frames

def predict(model, loader, device):
    predictions = []
    actual_strikes = []
    frame_numbers = []
    # No gradient needed for prediction, only inference
    with torch.no_grad():
        for data in loader:
            keypoints = data['keypoints'].to(device)
            labels = data['labels'].cpu().numpy()
            frames = data['Frame Number']
            # Forward pass to get outputs from the model
            outputs = model(keypoints)
            # Select the class with the highest probability
            _, preds = torch.max(outputs, 1)
            # Collect predictions, actual labels, and frame numbers
            predictions.extend(preds.cpu().numpy())
            actual_strikes.extend(labels)
            frame_numbers.extend(frames)
    return predictions, actual_strikes, frame_numbers

def evaluate_accuracy_per_strike_type(predictions, actual_strikes):
    accuracies = {}
    # Calculate accuracy for each strike type individually
    for strike_id, strike_name in enumerate(STRIKE_TYPES):
        actual_array = (np.array(actual_strikes) == strike_id)
        pred_array = (np.array(predictions) == strike_id)
        # Only calculate accuracy if there are actual instances of the strike type
        if actual_array.any():
            accuracy = accuracy_score(actual_array, pred_array)
            accuracies[strike_name] = accuracy
    return accuracies

def save_predictions_to_csv(predictions, actual_strikes, frames, csv_output_path):
    # Open CSV file for writing predictions
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Number', 'Predicted Strike', 'Actual Strike'])
        # Write each prediction with the corresponding frame number and strike type
        for frame, pred, actual in zip(frames, predictions, actual_strikes):
            writer.writerow([frame, STRIKE_TYPES[pred], STRIKE_TYPES[actual]])

if __name__ == '__main__':
    # Setup the device (GPU or CPU) for model computations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Specify paths for the model, CSV input, annotations, and CSV output
    model_path = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Model/model_fold_5_epoch_50.pth'
    csv_file = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Test/GloryRingTest.csv'
    annotations = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Test/GloryRingTest.xml'
    csv_output_path = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Model Output/GloryRingTestOutput.csv'

    # Load and parse annotations from XML
    annotations_dict = parse_annotations(annotations)

    # Load data, perform predictions, and save them
    predictions, actual_strikes, frames = load_data_and_predict(
        model_path, csv_file, annotations_dict, 51, 128, 2, device
    )
    save_predictions_to_csv(predictions, actual_strikes, frames, csv_output_path)

    # Evaluate the accuracy for each type of strike and print the results
    accuracies = evaluate_accuracy_per_strike_type(predictions, actual_strikes)
    print("Accuracy per strike type:", accuracies)
