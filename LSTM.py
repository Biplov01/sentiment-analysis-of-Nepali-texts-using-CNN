import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import torch.nn.functional as F

# Define the Bidirectional LSTM model for sentiment analysis
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))

# Load the data
data = pd.read_csv('data.csv')  # Load your dataset here
X = data['text']  # Assuming 'text' column contains the text data
y = data['label']  # Assuming 'label' column contains the sentiment labels

# Perform label encoding for y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
max_seq_length = 100  # You can adjust this value based on your input text length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length, padding='post')

# Define the BiLSTM model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
hidden_dim = 128
output_dim = len(label_encoder.classes_)
n_layers = 2
bidirectional = True
dropout = 0.5

# Initialize the BiLSTM model
model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train_pad, dtype=torch.long)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
X_test_torch = torch.tensor(X_test_pad, dtype=torch.long)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

# Define batch size and create DataLoader
batch_size = 64
train_data = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

# Train the model
num_epochs = 5  # You can adjust the number of epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Print training accuracy after each epoch
    train_accuracy = correct / total

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_torch)
    _, predicted = torch.max(outputs, 1)
    testing_accuracy = accuracy_score(y_test, predicted.numpy())

# Print overall training and testing accuracy
print("Overall Training Accuracy:", train_accuracy)
print("Overall Testing Accuracy:", testing_accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predicted.numpy())
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_test, predicted.numpy(), target_names=label_encoder.classes_))

# Save the trained model
model_path = 'bilstm_model.h5'
torch.save(model.state_dict(), model_path)
print(f"Trained model saved to {model_path}")
