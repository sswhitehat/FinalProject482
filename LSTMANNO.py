import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader, Dataset
import numpy as np
import xml.etree.ElementTree as ET

strike_types = ['No Strike', 'Jab', 'Cross', 'Hook', 'Upper', 'Leg Kick', 'Body Kick', 'High Kick']

strike_type_to_strike_id = {
    'No Strike': 1,
    'Jab': 2,
    'Cross': 3,
    'Hook': 4,
    'Upper': 5,
    'Leg Kick': 6,
    'Body Kick': 7,
    'High Kick': 8,
}

strike_id_to_strike_type = {
    1: 'No Strike',
    2: 'Jab',
    3: 'Cross',
    4: 'Hook',
    5: 'Upper',
    6: 'Leg Kick',
    7: 'Body Kick',
    8: 'High Kick',
}


class HybridBoxingLSTM(nn.Module):
    def __init__(self, keypoints_input_size, cnn_features_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.num_layers = num_layers  # Ensure this attribute is properly defined
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=keypoints_input_size + cnn_features_size,
                            hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

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
    for track in root.findall('.//track'):
        label = track.get('label')
        if label in strike_types:
            for box in track.findall('.//box'):
                frame = int(box.get('frame'))
                annotations[frame] = label
    return annotations


annotations = parse_annotations(
    'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Annotations/RodtangvGoncalvesAnnotations.xml')
num_samples = 9973
num_classes = len(strike_types)
seq_len = 17
keypoints_dim = 34
cnn_features_dim = 1280
keypoints = torch.randn(num_samples, seq_len, keypoints_dim)
cnn_features = torch.randn(num_samples, seq_len, cnn_features_dim)
labels = torch.full((num_samples,), len(strike_types), dtype=torch.long)  # Default to 'No Strike'
# label_map = {strike: i for i, strike in enumerate(strike_types)}

# print(annotations)

for frame in range(num_samples):
    if frame in annotations.keys():
        labels[frame] = strike_type_to_strike_id[annotations[frame]]
    else:
        labels[frame] = 1

print(labels)


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridBoxingLSTM(keypoints_dim, cnn_features_dim, 128, 2, len(strike_types) + 1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lowered from 0.001 to 0.0001
# class_weights = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
loss_function = nn.CrossEntropyLoss()

assert not torch.isnan(keypoints).any() and not torch.isnan(cnn_features).any(), "Inputs contain NaNs"
assert not torch.isnan(labels).any(), "Labels contain NaNs"
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
for epoch in range(30):
    total_loss = 0
    for kp, cnn, lbl in DataLoader(StrikeDataset(keypoints, cnn_features, labels), batch_size=10, shuffle=True):
        kp, cnn, lbl = kp.to(device), cnn.to(device), lbl.to(device)
        optimizer.zero_grad()
        outputs = model(kp, cnn)
        loss = loss_function(outputs, lbl)
        if torch.isnan(loss):
            print("NaN detected in loss, skipping backprop")
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss}')

# Evaluation loop
strike_count = {key: 0 for key in strike_types}

# Evaluation loop
model.eval()

# Prepare to write results to CSV
with open('RodtangvGoncalvesValidation.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame Number', 'Predicted Strike', 'Actual Strike'])

    with torch.no_grad():
        correct_matches = 0
        total = 0
        frame_offset = 0
        for kp, cnn, lbl in eval_dataloader:
            kp, cnn, lbl = kp.to(device), cnn.to(device), lbl.to(device)
            outputs = model(kp, cnn)
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_matches += (predicted_labels == lbl).sum().item()
            total += lbl.size(0)

            for i, (pred, actual) in enumerate(zip(predicted_labels, lbl)):
                frame_number = frame_offset + i
                predicted_strike = strike_id_to_strike_type[pred.item()]
                actual_strike = strike_id_to_strike_type[actual.item()]
                writer.writerow([frame_number, predicted_strike, actual_strike])

            frame_offset += len(lbl)

    accuracy = correct_matches / total
    print(f'Evaluation Accuracy: {accuracy * 100:.2f}%')

# Print the tally of predicted strikes
print("Tally of predicted strikes:")
for strike, count in strike_count.items():
    print(f"{strike}: {count}")
