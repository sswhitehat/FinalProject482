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

# Constant Values
STRIKE_TYPES = ['No Strike', 'Jab', 'Cross', 'Hook', 'Upper', 'Leg Kick', 'Body Kick', 'High Kick']
NUM_CLASSES = len(STRIKE_TYPES)
STRIKE_TYPE_TO_ID = {name: i for i, name in enumerate(STRIKE_TYPES)}


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Initializes the EarlyStopping mechanism.
        Parameters:
            patience (int): The number of epochs to wait before stopping after detecting no improvement.
            min_delta (float): The minimum change in the monitored quantity to qualify as an improvement.
        """
        # Number of epochs with no improvement after which training will be stopped
        self.patience = patience
        # Minimum change in the monitored quantity to qualify as an improvement; i.e., an improvement is seen
        # if the loss decreases by more than min_delta
        self.min_delta = min_delta
        # Counter to keep track of the number of epochs since the last improvement
        self.counter = 0
        # Variable to store the best loss achieved so far
        self.best_loss = None
        # Flag to indicate whether early stopping should be executed
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call method that operates each time the validation loss is computed.
        Parameters:
            val_loss (float): The current value of the validation loss.
        """
        # If the best_loss has not been set or if the new validation loss is a significant improvement,
        # reset the best_loss and counter
        if self.best_loss is None:
            self.best_loss = val_loss  # Set the best loss to the current value
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss  # Update the best loss to the new, lower loss
            self.counter = 0  # Reset the improvement counter
        else:
            # If no significant improvement is seen, increment the counter
            self.counter += 1
            # If the counter reaches the set patience, set the flag for early stopping
            if self.counter >= self.patience:
                self.early_stop = True


class ValidationKeypointDataset(Dataset):
    def __init__(self, csv_file, annotations):
        """
        Initializes the ValidationKeypointDataset with data loaded from a CSV file and annotation mapping.
        Parameters:
            csv_file (str): Path to the CSV file containing the keypoints data.
            annotations (dict): Dictionary mapping frame numbers to strike labels.
        """
        # Load data from the specified CSV file
        self.data_frame = pd.read_csv(csv_file)
        # Store the annotations dictionary which maps frame numbers to strike types
        self.annotations = annotations
        # Extract the names of columns that contain 'keypoint' in their header for filtering keypoint data
        self.keypoint_columns = [col for col in self.data_frame.columns if 'keypoint' in col]

        # Verify that the CSV contains all required keypoint columns
        if not set(self.keypoint_columns).issubset(set(self.data_frame.columns)):
            raise ValueError("CSV file does not contain all required keypoint columns.")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        Parameters:
            idx (int): The index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the keypoints, label, frame number, and actual strike information.
        """
        # Access the row corresponding to the index from the dataframe
        row = self.data_frame.iloc[idx]
        # Extract keypoints from the row and convert to a 1D numpy float32 array
        keypoints = row[self.keypoint_columns].values.astype(np.float32).reshape(-1)
        # Retrieve the frame number from the row
        frame_number = row['Frame Number']
        # Retrieve the actual strike value from the row
        actual_strike = row['Actual Strike']
        # Determine the label for the frame using the annotations dictionary, default to 'No Strike' if not found
        label = self.annotations.get(frame_number, STRIKE_TYPE_TO_ID['No Strike'])
        # Convert keypoints to a Torch tensor
        keypoints = torch.from_numpy(keypoints)
        # Return the sample as a dictionary
        return {'keypoints': keypoints, 'labels': label, 'frame_number': frame_number, 'actual_strike': actual_strike}


class ValidationComparisonDataset(Dataset):
    """
    A dataset class for comparing predicted strikes against actual strikes from a CSV file.
    This dataset is used for validation purposes to assess the accuracy of predictions.
    """

    def __init__(self, csv_file):
        """
        Initializes the dataset with the data loaded from a specified CSV file.
        Parameters:
            csv_file (str): The path to the CSV file containing the data.
        """
        # Load data from the specified CSV file
        self.data_frame = pd.read_csv(csv_file)

        # Check if the required columns are present in the DataFrame
        required_columns = {'Frame Number', 'Predicted Strike', 'Actual Strike'}
        if not required_columns.issubset(self.data_frame.columns):
            raise ValueError("CSV file does not contain all required columns: " + str(required_columns))

    def __len__(self):
        """
        Returns the total number of entries in the dataset.
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset by index.
        Parameters:
            idx (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the frame number, predicted strike, and actual strike.
        """
        # Access the data row corresponding to the index
        row = self.data_frame.iloc[idx]

        # Extract data from the row and map strike types from string to a predefined ID
        # Ensure that STRIKE_TYPE_TO_ID is defined elsewhere in your code to map strike names to their respective IDs
        return {
            'frame_number': row['Frame Number'],
            'predicted_strike': STRIKE_TYPE_TO_ID[row['Predicted Strike']],
            'actual_strike': STRIKE_TYPE_TO_ID[row['Actual Strike']]
        }


def validate_without_inference(validation_loader):
    """
    Validates the accuracy of predicted strike types against actual strike types from a DataLoader.

    This function assumes that predictions are pre-made and stored in the DataLoader, allowing
    for a straightforward comparison without the need to run any model inference steps.

    Parameters:
        validation_loader (DataLoader): A DataLoader object containing the dataset to validate.
                                        It should provide a dictionary with keys 'predicted_strike'
                                        and 'actual_strike', where each entry is a batch of data.

    Returns:
        float: The accuracy of the predictions as the ratio of correct predictions to the total predictions.
    """
    # Initialize counters for correct predictions and total samples
    correct = 0
    total = 0

    # Loop over each batch provided by the validation loader
    for data in validation_loader:
        # Extract the predicted and actual strike data from the batch
        predicted_strikes = data['predicted_strike']  # These should be tensors of predicted strike IDs
        actual_strikes = data['actual_strike']  # These should be tensors of actual strike IDs

        # Calculate the number of correct predictions in the current batch and add to the total correct
        correct += (predicted_strikes == actual_strikes).sum().item()  # Convert to int and sum

        # Update the total number of predictions processed
        total += predicted_strikes.size(0)  # Count the total predictions in this batch

    # Calculate and return the overall accuracy as the ratio of correct predictions to total predictions
    return correct / total


class KeypointDataset(Dataset):
    """
    A PyTorch Dataset class for handling keypoint data for machine learning tasks.
    This dataset includes preprocessing steps like normalization of keypoints using StandardScaler.
    """

    def __init__(self, data_frame, annotations):
        """
        Initializes the KeypointDataset with a pandas DataFrame and annotations.

        Parameters:
            data_frame (DataFrame): The DataFrame containing keypoint data along with other relevant information.
            annotations (dict): A dictionary mapping frame IDs to labels (e.g., different types of strikes).
        """
        self.data_frame = data_frame  # Pandas DataFrame containing the keypoints and other data
        self.annotations = annotations  # Dictionary for mapping frame IDs to annotated labels
        self.scaler = StandardScaler()  # Initialize a scaler to normalize the keypoint data
        # Identify all columns in the DataFrame that include 'keypoint' in their column name
        self.keypoint_columns = [col for col in self.data_frame.columns if 'keypoint' in col]
        # Fit the scaler on the keypoint data converted to float32 for consistency
        self.scaler.fit(self.data_frame[self.keypoint_columns].astype(np.float32))

    def __len__(self):
        """
        Returns the total number of entries in the dataset.
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset by index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the normalized keypoints as a tensor, the label as a tensor,
                  and the frame number.
        """
        # Access the data row corresponding to the index
        row = self.data_frame.iloc[idx]
        # Extract keypoints from the row, transform them using the fitted scaler, and select the first element (transform returns a list)
        keypoints = self.scaler.transform([row[self.keypoint_columns].values.astype(np.float32)])[0]
        # Retrieve the frame ID from the row, converting it to an integer
        frame_id = int(row['frame_id'])  # Ensure your data frame includes 'frame_id' or adjust accordingly
        # Retrieve the label for the frame using the annotations dictionary, defaulting to 'No Strike' if not found
        label = self.annotations.get(frame_id, STRIKE_TYPE_TO_ID['No Strike'])
        # Return the keypoints as a float tensor, the label as a long tensor, and the frame number
        return {'keypoints': torch.from_numpy(keypoints).float(), 'labels': torch.tensor(label, dtype=torch.long),
                'Frame Number': frame_id}


class KickBoxingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        """
        Initialize the HybridBoxingLSTM model with given parameters.
        Parameters:
            input_size (int): The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int): Number of recurrent layers in the LSTM
            dropout_rate (float): If non-zero, we introduce a Dropout layer on the outputs of each
            individual LSTM layer which is the last layer, with the dropout probability equal to the dropout_rate
        """
        super().__init__()
        # Define the LSTM layer with specified input size, hidden size, and number of layers
        # `batch_first=True` indicates that the input tensors will have a shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        # Define a fully connected layer to map the LSTM output to the number of classes
        self.fc = nn.Linear(hidden_size, NUM_CLASSES)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Parameters:
            x is our Tensor: Input data tensor, expected shape (batch, seq_length, input_size)
        Returns:
            Tensor: Output tensor after passing through LSTM, dropout, and fully connected layer
        """
        # Convert input to float32
        x = x.float()
        # Ensure input tensor is 3D (batch, 1, input_size) if currently 2D
        if x.dim() == 2:
            x = x.unsqueeze(1)
            # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, dtype=torch.float32).to(x.device)
        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Apply dropout on the outputs of the LSTM
        out = self.dropout(out)
        # Apply the fully connected layer on the last time step output
        return self.fc(out[:, -1, :])


def parse_annotations(xml_file):
    """
    Parses an XML file containing annotations for frames, typically used in datasets for image or video processing tasks.

    The XML structure is expected to have multiple 'track' elements, each representing a sequence of frames
    with a specific label (e.g., different types of actions or strikes). Each 'track' contains multiple 'box'
    elements, each of which corresponds to a frame.

    Parameters:
        xml_file (str): Path to the XML file containing frame annotations.

    Returns:
        dict: A dictionary mapping frame IDs to their corresponding labels, converted to numeric IDs.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    # Get the root of the XML tree
    root = tree.getroot()

    # Initialize a dictionary to store annotations
    annotations = {}

    # Iterate over each 'track' element in the XML
    for track in root.findall('.//track'):
        # Get the label for the current track
        label = track.get('label')
        # Extract frame IDs from all 'box' elements within the track
        frame_ids = [int(box.get('frame')) for box in track.findall('.//box')]
        # Map each frame ID to its corresponding label, using a predefined dictionary to convert labels to numeric IDs
        for frame_id in frame_ids:
            annotations[frame_id] = STRIKE_TYPE_TO_ID.get(label, STRIKE_TYPE_TO_ID['No Strike'])

    # Return the dictionary containing the mapped frame IDs and their corresponding labels
    return annotations


def validate_model(model, validation_loader, device):
    """
    Evaluates a machine learning model's performance on a validation dataset using multiple metrics.

    This function uses the provided validation loader to fetch batches of data, performs predictions
    using the model, and calculates various performance metrics including accuracy, precision,
    recall, F1 score, and a confusion matrix.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        validation_loader (DataLoader): A DataLoader that provides batches of validation data.
        device (torch.device): The device (e.g., "cpu" or "cuda") on which the model and data should be loaded.

    Returns:
        tuple: A tuple containing the accuracy, precision, recall, F1 score, and confusion matrix of the model on the validation set.
    """
    # Set the model to evaluation mode. This changes the behavior of some layers like BatchNorm and Dropout.
    model.eval()

    # Lists to store true and predicted labels for calculating metrics
    y_true, y_pred = [], []

    # Ensure no gradients are computed to save memory and computations
    with torch.no_grad():
        for data in validation_loader:
            # Transfer keypoints and labels to the specified device
            keypoints = data['keypoints'].to(device)
            labels = data['labels'].to(device)

            # Generate model predictions
            outputs = model(keypoints)
            # Extract the class with the highest probability
            _, predicted = torch.max(outputs.data, 1)

            # Append predictions and actual labels to lists, transferring them back to CPU
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calculate accuracy, precision, recall, F1 score, and confusion matrix using scikit-learn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Return all metrics
    return accuracy, precision, recall, f1, conf_matrix


# Suppress warnings that might occur during type conversions, commonly with scikit-learn
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def train_with_cross_validation(csv_file, xml_file, model_save_dir, device, input_size, num_layers, k_folds=5):
    """
    Trains a model using K-fold cross-validation on data specified in a CSV file and annotations in an XML file.
    The model is saved in the specified directory if it achieves an accuracy threshold during training.

    Parameters:
        csv_file (str): Path to the CSV file containing data.
        xml_file (str): Path to the XML file containing annotations.
        model_save_dir (str): Directory where trained models will be saved.
        device (torch.device): The device (CPU or GPU) on which to perform training.
        input_size (int): The number of input features for the model.
        num_layers (int): The number of layers in the LSTM model.
        k_folds (int): The number of folds to use for K-fold cross-validation.
    """
    # Load the data from a CSV file
    data = pd.read_csv(csv_file)
    # Check if the loaded data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data should be a pandas DataFrame.")

    print(f"Data loaded successfully, DataFrame shape: {data.shape}")

    # Parse annotations from the XML file
    annotations = parse_annotations(xml_file)

    # Initialize KFold object with specified number of splits, shuffling enabled and a fixed random seed for reproducibility
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Iterate over each fold, training and validating the model
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"Training fold {fold + 1}/{k_folds}")

        # Separate the data into training and validation datasets using the indices provided by KFold
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        # Initialize datasets and dataloaders for training and validation
        train_dataset = KeypointDataset(train_data, annotations)
        val_dataset = KeypointDataset(val_data, annotations)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

        # Create the model and move it to the specified device
        model = KickBoxingLSTM(input_size, 128, num_layers).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        loss_function = CrossEntropyLoss()

        # Perform the training loop
        for epoch in range(50):  # Modify the number of epochs if needed
            model.train()  # Set the model to training mode
            for batch in train_loader:
                keypoints = batch['keypoints'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad()  # Zero the gradients to prevent accumulation
                outputs = model(keypoints)
                loss = loss_function(outputs, labels)
                loss.backward()  # Perform backpropagation
                optimizer.step()  # Update model parameters

            # Validation loop to evaluate the model
            model.eval()  # Set the model to evaluation mode
            correct, total = 0, 0
            with torch.no_grad():  # Disable gradient calculation for efficiency
                for batch in val_loader:
                    keypoints = batch['keypoints'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(keypoints)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"Fold {fold + 1}, Epoch {epoch + 1}: Validation Accuracy = {accuracy:.4f}")

            # Save the model if it meets a specified accuracy threshold
            if accuracy > 0.90:
                model_path = os.path.join(model_save_dir, f'model_fold_{fold + 1}_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Model saved for fold {fold + 1}, epoch {epoch + 1}.")

    print("Training completed for all folds.")  # Indicate the end of the cross-validation training


def validate_by_strike_type(loader, model, device):
    """
    Validates a model by calculating the accuracy for each strike type individually.
    This function processes batches of keypoints from the loader and uses the model to predict strikes.
    It then compares these predictions to the actual labels to compute the accuracy for each type of strike.

    Parameters:
        loader (DataLoader): DataLoader containing the dataset to validate. The dataset should provide
                             dictionaries with 'keypoints' and 'labels'.
        model (torch.nn.Module): The trained model to be validated.
        device (torch.device): The device on which the model and data should be processed.

    Returns:
        dict: A dictionary where the keys are the names of the strikes and the values are the accuracy scores for
              each strike type.
    """
    # Set the model to evaluation mode to turn off dropout, batch normalization etc. during validation
    model.eval()

    # Lists to store predictions and labels for later analysis
    all_preds, all_labels = [], []

    # Disable gradient calculations for validation to save memory and computations
    with torch.no_grad():
        for data in loader:
            # Transfer the keypoints and labels to the specified device (e.g., GPU)
            keypoints = data['keypoints'].to(device)
            labels = data['labels'].to(device)

            # Perform model inference to get outputs
            outputs = model(keypoints)
            # Determine the predicted class by finding the index with the highest score in the output logits
            _, preds = torch.max(outputs, 1)

            # Collect all predictions and actual labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Dictionary to store the accuracy for each strike type
    accuracies = {}

    # Calculate accuracy for each strike type
    for idx, strike in enumerate(STRIKE_TYPES):
        # Create boolean arrays indicating where the specific strike type is the true label or the prediction
        specific_labels = (np.array(all_labels) == idx)
        specific_preds = (np.array(all_preds) == idx)

        # Only calculate accuracy if there are instances of this strike type in the true labels
        if specific_labels.sum() > 0:
            accuracies[strike] = accuracy_score(specific_labels, specific_preds)

    # Return the dictionary containing accuracies for each strike type
    return accuracies


def compute_class_weights(labels):
    """
    Computes class weights inversely proportional to the frequency of each class in the given labels.
    This method is useful for training on imbalanced datasets, as it allows minority classes to have a greater influence on the model.

    Parameters:
        labels (array-like): An array of integer class labels.

    Returns:
        np.ndarray: An array containing the computed weights for each class.
    """
    # Count the number of occurrences of each class in the dataset
    label_counts = np.bincount(labels, minlength=NUM_CLASSES)

    # Compute the inverse of each count to determine class weights
    class_weights = 1. / label_counts

    # If a class does not appear in the dataset (count is zero), set its weight to zero to avoid division by zero
    class_weights[label_counts == 0] = 0

    # Return the computed class weights
    return class_weights


def compare_with_validation_data(validation_csv, predictions, model_save_dir, epoch):
    """
    Compares predictions from a model with actual data in a validation dataset and saves the comparison
    to a CSV file. This function is useful for evaluating model predictions and performing error analysis.

    Parameters:
        validation_csv (str): Path to the CSV file containing validation data.
        predictions (list): List of model predictions (indices corresponding to classes).
        model_save_dir (str): Directory where the comparison result CSV will be saved.
        epoch (int): Current epoch number, used to label the output file.

    """
    # Load validation data from the specified CSV file
    validation_data = pd.read_csv(validation_csv)

    # Construct the path for the new CSV file that will store the comparison results
    prediction_file_path = os.path.join(model_save_dir, f"validation_comparison_epoch_{epoch}.csv")

    # Open the file for writing and create a CSV writer object
    with open(prediction_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['Frame Number', 'Model Prediction', 'Actual Strike'])

        # Iterate over the predictions and the corresponding rows in the validation dataset
        for i, pred in enumerate(predictions):
            # Ensure there is a corresponding row in the validation data for each prediction
            if i < len(validation_data):
                frame_number = validation_data.iloc[i]['Frame Number']
                actual_strike = validation_data.iloc[i]['Actual Strike']
                # Map the numerical prediction back to the corresponding strike type
                predicted_strike = STRIKE_TYPES[pred]
                # Write the comparison data to the CSV
                writer.writerow([frame_number, predicted_strike, actual_strike])


def make_predictions(model, loader, device):
    """
    Executes a model on data provided by a DataLoader and gathers the predictions.
    This function is typically used for generating predictions from a trained model in a testing
    or production environment where labels are not needed or used.

    Parameters:
        model (torch.nn.Module): The trained model to be used for predictions.
        loader (DataLoader): DataLoader containing the dataset over which predictions are to be made.
        device (torch.device): The device (e.g., "cpu" or "cuda") on which the model will operate.

    Returns:
        list: A list containing the predicted class indices for all the samples in the loader.
    """
    # Set the model to evaluation mode to deactivate dropout and other training-specific behaviors
    model.eval()

    # List to store the predictions
    predictions = []

    # Disable gradient calculations as they are not needed for inference, which saves memory and computations
    with torch.no_grad():
        for data in loader:
            # Move the keypoints data to the specified device (e.g., GPU)
            keypoints = data['keypoints'].to(device)

            # Perform the forward pass to get outputs from the model
            outputs = model(keypoints)

            # Extract the indices of the maximum values along the predicted output, which represent the class
            # predictions
            _, preds = torch.max(outputs, 1)

            # Append the predictions to the list after moving them back to the CPU and converting to numpy array
            predictions.extend(preds.cpu().numpy())

    # Return the list of predictions
    return predictions

def save_predictions_to_file(loader, predictions, filepath):
    """
    Saves predictions to a CSV file. This function is useful for storing model predictions in a structured format,
    which can be used for later analysis or for providing outputs in a production environment.

    Parameters:
        loader (DataLoader): DataLoader that was used to generate the predictions. This parameter is included for
                             potential future use where frame numbers or other identifiers might need to be extracted
                             directly from the DataLoader.
        predictions (list): List of integer indices representing the predicted class for each frame or sample.
        filepath (str): Path to the file where the predictions will be saved.

    Notes:
        - The frame number is assumed to be sequential and starting from zero, corresponding to the order of the samples
          in the DataLoader. If frame numbers need to be extracted differently, the function may require adjustments.
    """
    # Open the file at the specified path with write permissions and ensure it handles new lines appropriately for CSV
    with open(filepath, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Write the header row to the CSV file
        writer.writerow(['Frame Number', 'Predicted Strike'])

        # Iterate through each prediction and its corresponding index
        for i, pred in enumerate(predictions):
            # Frame number is assumed to be sequential starting at 0; modify if actual frame numbers are available
            frame_number = i
            # Convert the prediction index to the corresponding strike type using a predefined list or dictionary
            predicted_strike = STRIKE_TYPES[pred]
            # Write the frame number and predicted strike to the file
            writer.writerow([frame_number, predicted_strike])


def compare_predictions(validation_data, predictions, filepath):
    """
    Compares predictions made by a model against the actual data from a validation set and saves the comparison
    to a CSV file. This function is useful for reviewing prediction accuracy, performing detailed error analysis,
    and documenting results for further assessment.

    Parameters:
        validation_data (iterable): An iterable (e.g., list or DataLoader) containing dictionaries where each dictionary
                                    includes 'frame_number' and 'actual_strike' corresponding to each sample.
        predictions (list): List of predicted class indices from the model.
        filepath (str): Path to the file where the comparison results will be saved.

    Notes:
        - The function expects 'validation_data' to be in the same order as 'predictions'.
        - Each entry in 'validation_data' should have a 'frame_number' and 'actual_strike' key.
        - Predictions are converted from indices to more descriptive names using a predefined list or dictionary,
          `STRIKE_TYPES`.
    """
    # Open the specified file for writing; ensure newline behavior is handled correctly for different environments
    with open(filepath, 'w', newline='') as file:
        # Create a CSV writer object to handle the writing of rows
        writer = csv.writer(file)
        # Write the header of the CSV file to define the columns
        writer.writerow(['Frame Number', 'Predicted Strike', 'Actual Strike'])

        # Loop through each set of validation data and corresponding prediction
        for data, pred in zip(validation_data, predictions):
            # Extract the frame number and actual strike information from the validation data
            frame_number = data['frame_number']
            actual_strike = data['actual_strike']
            # Convert the numeric prediction into a more meaningful strike type using the `STRIKE_TYPES` dictionary
            predicted_strike = STRIKE_TYPES[pred]
            # Write the frame number, predicted strike, and actual strike to the CSV file
            writer.writerow([frame_number, predicted_strike, actual_strike])


# Function to write validation results to CSV file
def write_validation_results(loader, model, device, filepath):
    """
    Evaluates a model using data from a DataLoader and writes the predicted and actual strikes to a CSV file.
    This function is useful for model evaluation, particularly in validating the accuracy of predictions and
    documenting the results for further analysis.

    Parameters:
        loader (DataLoader): DataLoader containing the validation dataset, which provides batches of keypoints and
                             actual strikes.
        model (torch.nn.Module): The trained model to evaluate.
        device (torch.device): The device (e.g., GPU or CPU) where the model will perform the computations.
        filepath (str): Path to the CSV file where the results will be saved.

    Notes:
        - The 'keypoints' from each batch in the loader are used as input to the model.
        - The 'actual_strike' values are expected to be in the same batch dictionary as the 'keypoints'.
        - Predictions are mapped from numerical IDs back to strike names using an inverse dictionary
          (`STRIKE_TYPE_TO_ID.inverse`), which should be predefined.
        - Frame numbers are assumed to start from 0 and increment for each sample processed.
    """
    # Set the model to evaluation mode, which deactivates dropout and other training-specific layers
    model.eval()

    # Open the file with the specified path and ensure newline characters are handled correctly
    with open(filepath, 'w', newline='') as file:
        # Create a CSV writer to write to the file
        writer = csv.writer(file)
        # Write the header row to the CSV file
        writer.writerow(['Frame Number', 'Predicted Strike', 'Actual Strike'])

        # Initialize the frame number counter
        frame_number = 0

        # Disable gradient computations for more efficient memory use during inference
        with torch.no_grad():
            # Iterate over each batch provided by the DataLoader
            for data in loader:
                # Move the keypoints data to the specified computing device
                keypoints = data['keypoints'].to(device)
                # Get model outputs (logits) for the keypoints
                outputs = model(keypoints)
                # Determine the predicted labels by finding the max logit for each example
                _, predicted_labels = torch.max(outputs, 1)

                # Iterate over each actual label and its corresponding predicted label in the batch
                for actual, pred in zip(data['actual_strike'], predicted_labels):
                    # Map the predicted label index back to a strike name using an inverse dictionary
                    predicted_strike = STRIKE_TYPE_TO_ID.inverse[pred.item()]
                    # Write the frame number, predicted strike, and actual strike to the CSV
                    writer.writerow([frame_number, predicted_strike, actual])
                    # Increment the frame number for each sample processed
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
