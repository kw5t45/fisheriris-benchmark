import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
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
    def __init__(self, hidden_neurons=10):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_neurons),
            nn.Linear(hidden_neurons, 3)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------
# 3. Initialize model, loss, optimizer
# -------------------------------
model = NeuralNetwork(hidden_neurons=10)
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

    if (epoch + 1) % 20 == 0:
        with torch.no_grad():  # no gradients for evaluation
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test_onehot)

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# -------------------------------
# 4️⃣.5  Training loop of Extreme Learning Machine
# -------------------------------
            # ua to grapsei o kapas


# -------------------------------
# 5️⃣ Simple prediction
# -------------------------------
# Predict on the first test sample
test_sample = X_test[0].unsqueeze(0)  # add batch dimension
pred = model(test_sample)
print(test_sample, pred)
# Convert output to class
predicted_class = torch.argmax(pred, dim=1).item()
true_class = torch.argmax(y_test_onehot[0]).item()

print("\nPredicted class:", label_encoder.inverse_transform([predicted_class])[0])
print("True class:     ", label_encoder.inverse_transform([true_class])[0])

