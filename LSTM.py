import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# Constants
STRIKE_TYPES = ['No Strike', 'Jab', 'Cross', 'Hook', 'Upper', 'Leg Kick', 'Body Kick', 'High Kick']
NUM_CLASSES = len(STRIKE_TYPES)
STRIKE_TYPE_TO_ID = {name: i for i, name in enumerate(STRIKE_TYPES)}


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ValidationKeypointDataset(Dataset):
    def __init__(self, csv_file, annotations):
        self.data_frame = pd.read_csv(csv_file)
        self.annotations = annotations
        self.keypoint_columns = [col for col in self.data_frame.columns if 'keypoint' in col]

        # Check if necessary columns are present
        if not set(self.keypoint_columns).issubset(set(self.data_frame.columns)):
            raise ValueError("CSV file does not contain all required keypoint columns.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        keypoints = row[self.keypoint_columns].values.astype(np.float32).reshape(-1)
        frame_number = row['Frame Number']
        actual_strike = row['Actual Strike']
        label = self.annotations.get(frame_number, STRIKE_TYPE_TO_ID['No Strike'])
        keypoints = torch.from_numpy(keypoints)
        return {'keypoints': keypoints, 'labels': label, 'frame_number': frame_number, 'actual_strike': actual_strike}


class ValidationComparisonDataset(Dataset):
    """ Dataset for validation comparison, no keypoints needed. """

    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        if not {'Frame Number', 'Predicted Strike', 'Actual Strike'}.issubset(self.data_frame.columns):
            raise ValueError("CSV file does not contain all required columns.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        return {
            'frame_number': row['Frame Number'],
            'predicted_strike': STRIKE_TYPE_TO_ID[row['Predicted Strike']],
            'actual_strike': STRIKE_TYPE_TO_ID[row['Actual Strike']]
        }


def validate_without_inference(validation_loader):
    correct = 0
    total = 0
    for data in validation_loader:
        predicted_strikes = data['predicted_strike']  # These are tensors, not single values
        actual_strikes = data['actual_strike']
        correct += (predicted_strikes == actual_strikes).sum().item()  # Correctly count matches
        total += predicted_strikes.size(0)  # Total number of predictions in this batch
    return correct / total


class KeypointDataset(Dataset):
    def __init__(self, data_frame, annotations):
        self.data_frame = data_frame
        self.annotations = annotations
        self.scaler = StandardScaler()
        self.keypoint_columns = [col for col in self.data_frame.columns if 'keypoint' in col]
        self.scaler.fit(self.data_frame[self.keypoint_columns].astype(np.float32))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        keypoints = self.scaler.transform([row[self.keypoint_columns].values.astype(np.float32)])[0]
        frame_id = int(row['frame_id'])  # Ensure your CSV has this column or adjust accordingly
        label = self.annotations.get(frame_id, STRIKE_TYPE_TO_ID['No Strike'])
        return {'keypoints': torch.from_numpy(keypoints).float(), 'labels': torch.tensor(label, dtype=torch.long),
                'Frame Number': frame_id}


class HybridBoxingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, NUM_CLASSES)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.float()  # Ensure input tensor is float32
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Ensure input is 3D as expected by LSTM
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, dtype=torch.float32).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        return self.fc(out[:, -1, :])  # Predict based on the last output

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = {}
    for track in root.findall('.//track'):
        label = track.get('label')
        frame_ids = [int(box.get('frame')) for box in track.findall('.//box')]
        for frame_id in frame_ids:
            annotations[frame_id] = STRIKE_TYPE_TO_ID.get(label, STRIKE_TYPE_TO_ID['No Strike'])
    return annotations


def validate_model(model, validation_loader, device):
    """ Function to evaluate the model on the validation set """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in validation_loader:
            keypoints = data['keypoints'].to(device)
            labels = data['labels'].to(device)
            outputs = model(keypoints)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, conf_matrix


warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def train_with_cross_validation(csv_file, xml_file, model_save_dir, device, input_size, num_layers, k_folds=5):
    # Load the data
    data = pd.read_csv(csv_file)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data should be a pandas DataFrame.")

    print(f"Data loaded successfully, DataFrame shape: {data.shape}")

    # Parse annotations
    annotations = parse_annotations(xml_file)

    # Set up K-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Train and validate across folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"Training fold {fold + 1}/{k_folds}")

        # Extract train and validation data using indices
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        # Setup datasets and dataloaders
        train_dataset = KeypointDataset(train_data, annotations)
        val_dataset = KeypointDataset(val_data, annotations)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

        # Initialize model, optimizer, and loss function
        model = HybridBoxingLSTM(input_size, 128, num_layers).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        loss_function = CrossEntropyLoss()

        # Training loop
        for epoch in range(50):  # Use more epochs as needed
            model.train()
            for batch in train_loader:
                keypoints = batch['keypoints'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad()
                outputs = model(keypoints)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    keypoints = batch['keypoints'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(keypoints)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"Fold {fold + 1}, Epoch {epoch + 1}: Validation Accuracy = {accuracy:.4f}")

            # Save model if accuracy threshold is met
            if accuracy > 0.90:
                model_path = os.path.join(model_save_dir, f'model_fold_{fold + 1}_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Model saved for fold {fold + 1}, epoch {epoch + 1}.")

    print("Training completed for all folds.")

def validate_by_strike_type(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            keypoints = data['keypoints'].to(device)
            labels = data['labels'].to(device)
            outputs = model(keypoints)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracies = {}
    for idx, strike in enumerate(STRIKE_TYPES):
        specific_labels = (np.array(all_labels) == idx)
        specific_preds = (np.array(all_preds) == idx)
        if specific_labels.sum() > 0:
            accuracies[strike] = accuracy_score(specific_labels, specific_preds)
    return accuracies


def compute_class_weights(labels):
    label_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1. / label_counts
    class_weights[label_counts == 0] = 0  # handle classes with zero samples gracefully
    return class_weights


def compare_with_validation_data(validation_csv, predictions, model_save_dir, epoch):
    # Load validation data
    validation_data = pd.read_csv(validation_csv)
    prediction_file_path = os.path.join(model_save_dir, f"validation_comparison_epoch_{epoch}.csv")

    with open(prediction_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'Model Prediction', 'Actual Strike'])
        for i, pred in enumerate(predictions):
            if i < len(validation_data):
                frame_number = validation_data.iloc[i]['Frame Number']
                actual_strike = validation_data.iloc[i]['Actual Strike']
                predicted_strike = STRIKE_TYPES[pred]
                writer.writerow([frame_number, predicted_strike, actual_strike])


def make_predictions(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in loader:
            keypoints = data['keypoints'].to(device)
            outputs = model(keypoints)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions


def save_predictions_to_file(loader, predictions, filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'Predicted Strike'])
        for i, pred in enumerate(predictions):
            frame_number = i  # Frame number can be assumed sequential or extracted differently if needed
            predicted_strike = STRIKE_TYPES[pred]
            writer.writerow([frame_number, predicted_strike])


def compare_predictions(validation_data, predictions, filepath):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'Predicted Strike', 'Actual Strike'])
        for data, pred in zip(validation_data, predictions):
            frame_number = data['frame_number']
            actual_strike = data['actual_strike']
            predicted_strike = STRIKE_TYPES[pred]
            writer.writerow([frame_number, predicted_strike, actual_strike])


# Function to write validation results to CSV file
def write_validation_results(loader, model, device, filepath):
    model.eval()
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'Predicted Strike', 'Actual Strike'])
        frame_number = 0
        with torch.no_grad():
            for data in loader:
                keypoints = data['keypoints'].to(device)
                outputs = model(keypoints)
                _, predicted_labels = torch.max(outputs, 1)

                for actual, pred in zip(data['actual_strike'], predicted_labels):
                    predicted_strike = STRIKE_TYPE_TO_ID.inverse[pred.item()]
                    writer.writerow([frame_number, predicted_strike, actual])
                    frame_number += 1


# Main Function
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 51  # Number of input features (e.g., number of keypoints * coordinates)
    num_layers = 2  # Number of LSTM layers
    num_classes = 8  # Number of classes (number of different strike types)
    model_save_dir = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Model'  # Directory to save the trained models

    # Lists of CSV, XML, and Validation CSV files
    csv_files = [
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Keypoint Files/SuperbonvPetrosyanKeypoints.csv',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Keypoint Files/SlugfestVideo.csv',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Keypoint Files/Trimmed_Haggerty_vs_Mongkolpetch_No_CommentaryKeypoints.csv',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Keypoint Files/Trimmed_Haggerty_vs_Naito_No_CommentaryKeypoints.csv',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Keypoint Files/Trimmed_Rodtang_vs_Goncalves_No_CommentaryKeypoints.csv'
    ]

    xml_files = [
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Annotations/SuperbonvPetrysan.xml',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Annotations/slugfestannotations.xml',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Annotations/HaggertyvMongkolpetchAnnotations.xml',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Annotations/HaggertyvNaitoAnnotations.xml',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Annotations/RodtangvGoncalvesAnnotations.xml'
    ]

    validation_csv_files = [
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Validation/SuperbonvPetryosanValidation.csv',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Validation/SlugFestValidation.csv',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Validation/HaggertyvMongkolpetchValidation.csv',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Validation/HaggertyvNaitoValidation.csv',
        'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Validation/RodtangvGoncalvesValidation.csv'
    ]

    # Call the training function with all necessary parameters
    for csv_file, xml_file in zip(csv_files, xml_files):
        print(f"Starting training for {csv_file} using annotations from {xml_file}")
        train_with_cross_validation(csv_file, xml_file, model_save_dir, device, input_size, num_layers)
