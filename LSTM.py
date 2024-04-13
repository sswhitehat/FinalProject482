import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from torchvision import models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import json

# Simulating the loading of the raw label data
raw_label_data = '''
[
  {"name": "Jab", "id": 2283330, "color": "#ff6037", "type": "rectangle", "attributes": []},
  {"name": "Cross", "id": 2296104, "color": "#3df53d", "type": "rectangle", "attributes": []},
  {"name": "F1", "id": 2296105, "color": "#61a12e", "type": "any", "attributes": []},
  {"name": "F2", "id": 2296106, "color": "#a519c9", "type": "any", "attributes": []},
  {"name": "Ref", "id": 2296107, "color": "#a535a2", "type": "any", "attributes": []},
  {"name": "Hook", "id": 2296108, "color": "#31c414", "type": "any", "attributes": []},
  {"name": "Upper", "id": 2296109, "color": "#a92616", "type": "any", "attributes": []},
  {"name": "Leg Kick", "id": 2296110, "color": "#5cd611", "type": "any", "attributes": []},
  {"name": "Body Kick", "id": 2296111, "color": "#ff6a4d", "type": "any", "attributes": []},
  {"name": "High Kick", "id": 2296112, "color": "#ea3e5b", "type": "any", "attributes": []}
]
'''

# Debug function to check the sizes
def print_tensor_sizes(keypoints, cnn_features):
    print("Keypoints tensor size:", keypoints.shape)
    print("CNN features tensor size:", cnn_features.shape)
    combined_input_size = keypoints.shape[2] + cnn_features.shape[2]
    print("Combined input size should be:", combined_input_size)
    return combined_input_size

class HybridBoxingLSTM(nn.Module):
    def __init__(self, keypoints_input_size, cnn_features_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(keypoints_input_size + cnn_features_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, keypoints, cnn_features):
        combined_input = torch.cat((keypoints, cnn_features), dim=2)
        h0 = torch.zeros(self.num_layers, combined_input.size(0), self.hidden_size).to(combined_input.device)
        c0 = torch.zeros(self.num_layers, combined_input.size(0), self.hidden_size).to(combined_input.device)
        out, _ = self.lstm(combined_input, (h0, c0))
        return self.fc(out[:, -1, :])

def load_keypoints(csv_file):
    df = pd.read_csv(csv_file)
    return df[[f'keypoint_{i}_{xy}' for i in range(17) for xy in ['x', 'y']]].values.reshape(-1, 17, 2)

def create_label_mapping(labels):
    action_to_idx = {label['name']: idx for idx, label in enumerate(labels)}
    action_to_idx['Unknown'] = len(action_to_idx)  # Adding an 'Unknown' category
    return action_to_idx

def integrate_data(keypoints, annotations, action_to_idx):
    labels = []
    for i in range(len(keypoints)):
        frame_id = str(i)
        action = annotations.get(frame_id, 'Unknown')
        labels.append(action_to_idx.get(action, action_to_idx['Unknown']))
    return keypoints, np.array(labels)

def load_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = {}
    for frame in root.findall('frame'):
        frame_id = frame.get('id')
        action = frame.find('action').text if frame.find('action') is not None else 'Unknown'
        annotations[frame_id] = action
    return annotations

labels = json.loads(raw_label_data)
action_to_idx = create_label_mapping(labels)
annotations = load_annotations('annotations.xml')
keypoints = load_keypoints('keypoints_data.csv')
keypoints, labels = integrate_data(keypoints, annotations, action_to_idx)

class BoxingDataset(Dataset):
    def __init__(self, keypoints, labels):
        self.keypoints = torch.tensor(keypoints, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return self.keypoints[idx], self.labels[idx]

dataset = BoxingDataset(keypoints, labels)
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
mobilenet.eval()

def extract_cnn_features(frames):
    with torch.no_grad():
        return mobilenet(frames)

model = HybridBoxingLSTM(34 * 2, 1280, 128, 2, len(labels))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for keypoints_batch, labels_batch in data_loader:
        cnn_features_batch = torch.randn(keypoints_batch.shape[0], keypoints_batch.shape[1], 1280)
        outputs = model(keypoints_batch, cnn_features_batch)
        loss = loss_function(outputs, labels_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
