import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.linalg as linalg
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold


# dataset loading
dataset = pd.read_csv('IRIS.csv')

x_data = dataset.drop('species', axis=1)
y_labels = dataset['species']

# Encode species labels (string → number)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

# Normalize features for better training
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_data)

# Train-test split
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# adjustable mlp

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers ):
        """
        input_size: int — number of input features
        hidden_layers: list[int] — list with the number of neurons per hidden layer
        output_size: int — number of output neurons
        """
        super(NeuralNetwork, self).__init__()

        layers = []
        prev_size = 4

        for hidden_size in hidden_layers:
            if hidden_size > 0:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.Sigmoid())  # optional activation
                prev_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(prev_size, 3))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# mlp training

resLoss=[]
resTestLoss=[]

start = time.time()
for i in range(1,6):
    hLyeres = []
    for j in range(1,6):
        hLyeres.append(i)
        model = NeuralNetwork(hLyeres)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        fold_train_losses = []
        fold_val_losses = []

        # 🔹 K-Fold training
        for train_idx, val_idx in kf.split(x_scaled):
            X_train_fold = torch.tensor(x_scaled[train_idx], dtype=torch.float32)
            X_val_fold = torch.tensor(x_scaled[val_idx], dtype=torch.float32)

            y_train_fold = torch.tensor(pd.get_dummies(y_encoded[train_idx]).values, dtype=torch.float32)
            y_val_fold = torch.tensor(pd.get_dummies(y_encoded[val_idx]).values, dtype=torch.float32)

            # train the model
            for epoch in range(1000):
                outputs = model(X_train_fold)
                loss = criterion(outputs, y_train_fold)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # evaluate on validation fold
            with torch.no_grad():
                val_outputs = model(X_val_fold)
                val_loss = criterion(val_outputs, y_val_fold)

            fold_train_losses.append(loss.item())
            fold_val_losses.append(val_loss.item())

        # average fold losses
        resLoss.append(sum(fold_train_losses)/len(fold_train_losses))
        resTestLoss.append(sum(fold_val_losses)/len(fold_val_losses))

end = time.time()  # ⏱️ record end time
print(f"MLPs total training took {end - start:.4f} seconds")
print(resTestLoss)
# elm initialization
class ExtremeLearningMachine:
    def __init__(self, hidden_neurons, activation=torch.sigmoid):
        """
        input_size: number of input features
        hidden_neurons: number of neurons in hidden layer
        output_size: number of output neurons
        activation: activation function (default ReLU)
        """
        self.input_size = 4
        self.hidden_neurons = hidden_neurons
        self.output_size = 3
        self.activation = activation

        # Random weights and bias for input → hidden
        self.W = torch.randn(4, hidden_neurons)
        self.b = torch.randn(hidden_neurons)

        # Output weights (to be learned analytically)
        self.beta = None

    def _hidden_output(self, X):
        """Compute hidden layer output"""
        H = self.activation(X @ self.W + self.b)
        return H

    def fit(self, X, y):
        """
        Train ELM by computing output weights (beta)
        using Moore-Penrose pseudoinverse
        """
        H = self._hidden_output(X)
        # Compute pseudoinverse and solve: beta = pinv(H) * y
        self.beta = linalg.pinv(H) @ y

    def predict(self, X):
        """Make predictions"""
        H = self._hidden_output(X)
        return H @ self.beta



exlm_losses = []
num_classes = 3
kf = KFold(n_splits=5, shuffle=True, random_state=42)
criterion = nn.MSELoss()  # define once

exlm_start_time = time.time()

for hidden_neurons in range(1, 26):  # 1 to 25
    fold_losses = []  # store loss for each fold

    for train_index, val_index in kf.split(x_scaled):

        X_train_fold = torch.tensor(x_scaled[train_index], dtype=torch.float32)
        X_val_fold = torch.tensor(x_scaled[val_index], dtype=torch.float32)

        y_train_fold = torch.tensor(pd.get_dummies(y_encoded[train_index]).values, dtype=torch.float32)
        y_val_fold = torch.tensor(pd.get_dummies(y_encoded[val_index]).values, dtype=torch.float32)

        # initialize and train ELM on this fold
        elm = ExtremeLearningMachine(
            hidden_neurons=hidden_neurons,
            activation=torch.sigmoid
        )
        elm.fit(X_train_fold, y_train_fold)

        # predict and compute loss
        pred = elm.predict(X_val_fold)
        pred = torch.softmax(pred, dim=1)
        loss = criterion(pred, y_val_fold)

        fold_losses.append(loss.item())

    # average loss over all folds for this number of neurons
    exlm_losses.append(np.mean(fold_losses))

exlm_total_time = time.time() - exlm_start_time
print("Cross-validated ELM losses:", exlm_losses)
print("Total time:", exlm_total_time)

# Create a DataFrame
df = pd.DataFrame({
    'MLP Train Loss': resLoss,
    'MLP Test Loss': resTestLoss,
    'ELM Loss': exlm_losses
})

# Save to CSV
#df.to_csv('full.csv', index=False, float_format='%.3f')
print(df)