import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv
import torch.linalg as linalg


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler



#test
# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
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
X_train, X_test, y_train, y_test = train_test_split(
    x_scaled, y_encoded, test_size=0.2, random_state=42
)

# One-hot encode targets for MSELoss
y_train_onehot = torch.tensor(pd.get_dummies(y_train).values, dtype=torch.float32)
y_test_onehot = torch.tensor(pd.get_dummies(y_test).values, dtype=torch.float32)

# Convert inputs to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)




# -------------------------------
# 2 Define a simple MLP
# -------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_neurons2 ,hidden_neurons):
        layers = [nn.Linear(4, hidden_neurons)]

        if hidden_neurons2 > 0:
            layers.append(nn.Linear(hidden_neurons, hidden_neurons2))
            layers.append(nn.Linear(hidden_neurons2, 3))
        else:
            layers.append(nn.Linear(hidden_neurons, 3))

        self.model = nn.Sequential(*layers)
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_neurons),
            nn.Linear(hidden_neurons,hidden_neurons2),
            nn.Linear(hidden_neurons2, 3)
        )



    def forward(self, x):
        return self.model(x)

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
                layers.append(nn.ReLU())  # optional activation
                prev_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(prev_size, 3))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# -------------------------------
# 3. Initialize model, loss, optimizer
# -------------------------------

resLoss=[]
resTestLoss=[]
for i in range(1,6):
    hLyeres = []
    for j in range(1,6):
        hLyeres.append(i)
        print(hLyeres)
        model = NeuralNetwork(hLyeres)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)


        # -------------------------------
        # 4️⃣ Training loop
        # -------------------------------
        epochs = 400
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train_onehot)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        with torch.no_grad():  # no gradients for evaluation
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test_onehot)

            resLoss.append(loss.item())
            resTestLoss.append(test_loss.item())

            #print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# -------------------------------
# 4️⃣.5  Training loop of Extreme Learning Machine
# -------------------------------
class ExtremeLearningMachine:
    def __init__(self, hidden_neurons, activation=torch.relu):
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







# Create ELM with user-defined hidden neurons
exlm_losses=[]
for hidden_neurons in range(1,26):  # ← user can change this freely

    elm = ExtremeLearningMachine(hidden_neurons=hidden_neurons,
                                 activation=torch.relu)
    elm.fit(X_train, y_train_onehot)
    pred = elm.predict(X_test)
    pred = torch.softmax(pred, dim=1)
    criterion = nn.MSELoss()       # create the loss function
    loss = criterion(pred, y_test_onehot)   # compute the actual loss
    # print("Loss :",loss)

    exlm_losses.append(loss.item())

# Create a DataFrame
df = pd.DataFrame({
    'Result (Train) Loss': resLoss,

    'Test Loss': resTestLoss,

    'ExLM Loss': exlm_losses
})

# Save to CSV
df.to_csv('data.csv', index=False)
