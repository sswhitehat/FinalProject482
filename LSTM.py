import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import xml.etree.ElementTree as ET

strike_types = ['Jab', 'Cross', 'Hook', 'Upper', 'Leg Kick', 'Body Kick', 'High Kick','No Strike']  # Added 'No Strike' as a legitimate class

class HybridBoxingLSTM(nn.Module):
    def __init__(self, keypoints_input_size, cnn_features_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=keypoints_input_size + cnn_features_size,
                            hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, keypoints, cnn_features):
        combined_input = torch.cat((keypoints, cnn_features), dim=2)
        h0 = torch.zeros(self.num_layers, combined_input.size(0), self.hidden_size).to(combined_input.device)
        c0 = torch.zeros(self.num_layers, combined_input.size(0), self.hidden_size).to(combined_input.device)
        out, _ = self.lstm(combined_input, (h0, c0))
        return self.fc(out[:, -1, :])

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = {}
    non_strikes = ['F1', 'F2', 'Ref']  # Define non-strike labels

    for track in root.findall('.//track'):
        label = track.get('label')
        if label in strike_types:  # Only consider configured strike types
            for box in track.findall('.//box'):
                frame = int(box.get('frame'))
                annotations[frame] = label
        elif label in non_strikes:  # Handle non-strike labels if needed
            continue  # Skip non-strike labels or assign them a 'No Strike' class

    return annotations


annotations = parse_annotations('annotations.xml')

# Mock-up data for demonstration
num_samples = 250
seq_len = 17
keypoints_dim = 34
cnn_features_dim = 1280
keypoints = torch.randn(num_samples, seq_len, keypoints_dim)
cnn_features = torch.randn(num_samples, seq_len, cnn_features_dim)

num_classes = len(strike_types) + 1  # Add an extra class for 'No Strike'
labels = torch.full((num_samples,), num_classes-1, dtype=torch.long)  # Default to 'No Strike'
label_map = {strike: i for i, strike in enumerate(strike_types)}

# Create labels array with default 'No Strike' class index

for frame in range(min(num_samples, max(annotations.keys()))):  # Ensure we only go up to the highest frame annotated
    strike = annotations.get(frame)
    labels[frame] = label_map.get(strike, num_classes-1)  # Default to 'No Strike' if not found


print(labels)
print(cnn_features)
class StrikeDataset(Dataset):
    def __init__(self, keypoints, cnn_features, labels):
        self.keypoints = keypoints
        self.cnn_features = cnn_features
        self.labels = labels

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return self.keypoints[idx], self.cnn_features[idx], self.labels[idx]

# Create dataset and dataloader
dataset = StrikeDataset(keypoints, cnn_features, labels)
train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # Use shuffling during training
eval_dataloader = DataLoader(dataset, batch_size=10, shuffle=False)  # No shuffling during evaluation to maintain order

model = HybridBoxingLSTM(keypoints_dim, cnn_features_dim, 128, 2, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss(ignore_index=num_classes-1)  # Ignoring the 'No Strike' class

# Training loop
for epoch in range(20):  # Example: train for 3 epochs
    for kp, cnn, lbl in train_dataloader:
        optimizer.zero_grad()
        outputs = model(kp, cnn)
        valid_indices = lbl != num_classes-1  # Find indices where label is not 'No Strike'
        if valid_indices.any():  # Only compute loss where there are valid indices
            loss = loss_function(outputs[valid_indices], lbl[valid_indices])
            loss.backward()
            optimizer.step()
    print(f'Epoch {epoch + 1}: Loss {loss.item()}')

# Evaluation loop
with torch.no_grad():
    frame_offset = 0  # Initialize frame offset for tracking frames across batches
    for kp, cnn, lbl in eval_dataloader:
        predictions = model(kp, cnn)
        predicted_labels = torch.argmax(predictions, dim=1)
        for i, (pred, actual) in enumerate(zip(predicted_labels, lbl)):
            frame_number = frame_offset + i  # Compute global frame number
            predicted_strike = strike_types[pred] if pred < len(strike_types) else 'No Strike'
            actual_strike = strike_types[actual] if actual < len(strike_types) else 'No Strike'
            print(f"Evaluation - Frame {frame_number}: Predicted Strike: {predicted_strike}, Actual Strike: {actual_strike}")
        frame_offset += len(lbl)  # Update the frame offset by the batch size

