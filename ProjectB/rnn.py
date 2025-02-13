import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import re
from logisticregression import DatabaseLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = [self.tokenize(text, vocab, max_length) for text in texts]
        self.labels = labels

    def tokenize(self, text, vocab, max_length):
        text = re.sub(r'[^a-zA-Z]', ' ', text.lower()).split()
        tokens = [vocab.get(word, vocab['UNK']) for word in text]
        if len(tokens) < max_length:
            tokens += [vocab['PAD']] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

class StackedBiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, embedding_matrix, dropout=0.5):
        super(StackedBiRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = torch.max(output, dim=1)[0]  # Global max pooling
        output = self.dropout(output)
        logits = self.fc(output)
        return self.sigmoid(logits)  # Apply sigmoid to logits

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, rnn_type, pretrained=True, freeze=False, use_pooling=False):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.use_pooling = use_pooling
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        if self.use_pooling:
            output = torch.max(output, dim=1)[0]  # Global max pooling
        else:
            output = output[:, -1, :]  # Use the last hidden state
        logits = self.fc(output)
        return self.sigmoid(logits).squeeze()

def main():
    loader = DatabaseLoader()
    train_pos = loader.load_reviews(loader.TRAINING_POSITIVE)
    train_neg = loader.load_reviews(loader.TRAINING_NEGATIVE)
    test_pos = loader.load_reviews(loader.TEST_POSITIVE)
    test_neg = loader.load_reviews(loader.TEST_NEGATIVE)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    texts = train_pos + train_neg + test_pos + test_neg
    labels = [1] * len(train_pos) + [0] * len(train_neg) + [1] * len(test_pos) + [0] * len(test_neg)
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    vectorizer = CountVectorizer(max_features=10000)
    vectorizer.fit(X_train)
    custom_vocab = vectorizer.vocabulary_

    word2vec = api.load('word2vec-google-news-300')

    embedding_dim = 300
    average_embedding = np.mean(word2vec.vectors, axis=0)
    vocab = {'PAD': 0, 'UNK': 1}
    vocab.update({word: idx + 2 for idx, word in enumerate(custom_vocab)})
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    embedding_matrix[0] = np.zeros(embedding_dim)  # PAD token
    embedding_matrix[1] = average_embedding  # UNK token
    for word, idx in vocab.items():
        if word in word2vec:
            embedding_matrix[idx] = word2vec[word]
        elif idx > 1:
            embedding_matrix[idx] = average_embedding

    avg_length = int(np.mean([len(re.sub(r'[^a-zA-Z]', ' ', text.lower()).split()) for text in X_train]))
    train_dataset = TextDataset(X_train, y_train, vocab, avg_length)
    val_dataset = TextDataset(X_val, y_val, vocab, avg_length)
    test_dataset = TextDataset(X_test, y_test, vocab, avg_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    models = {
        'RNN': RNNModel(len(vocab), 300, 32, 1, 'RNN'),
        'GRU': RNNModel(len(vocab), 300, 32, 1, 'GRU'),
        'LSTM': RNNModel(len(vocab), 300, 32, 1, 'LSTM'),
        'LSTM_GMP': RNNModel(len(vocab), 300, 32, 1, 'LSTM', use_pooling=True),
        'StackedBiRNN': StackedBiRNN(len(vocab), 300, 32, 2, 1, embedding_matrix),  # 2 layers, bidirectional
        'RandomInit': RNNModel(len(vocab), 300, 32, 1, 'LSTM', pretrained=False, use_pooling=True),
        'FrozenWord2Vec': RNNModel(len(vocab), 300, 32, 1, 'LSTM', pretrained=True, freeze=True, use_pooling=True),
        'TrainableWord2Vec': RNNModel(len(vocab), 300, 32, 1, 'LSTM', pretrained=True, freeze=False, use_pooling=True)
    }

    results = {}
    epochs = 10

    # Train and evaluate models
    for name, model in models.items():
        model = model.to(device)  # Move model to the correct device
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"Training {name}...")
        train_losses, val_losses, best_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)
        results[name] = {'train_loss': train_losses, 'val_loss': val_losses}
        metrics = evaluate_model(best_model, test_loader, device)
        results[name].update(metrics)

    # Plot losses
    for name in models:
        plt.plot(results[name]['train_loss'], linestyle='--', label=f'{name} Train')
        plt.plot(results[name]['val_loss'], label=f'{name} Val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train/Validation Loss Comparison')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    # Print evaluation metrics
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (Class 0): {metrics['precision'][0]:.4f}, Precision (Class 1): {metrics['precision'][1]:.4f}")
        print(f"  Recall (Class 0): {metrics['recall'][0]:.4f}, Recall (Class 1): {metrics['recall'][1]:.4f}")
        print(f"  F1 (Class 0): {metrics['f1'][0]:.4f}, F1 (Class 1): {metrics['f1'][1]:.4f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}, Recall (Macro): {metrics['recall_macro']:.4f}, F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"  Precision (Micro): {metrics['precision_micro']:.4f}, Recall (Micro): {metrics['recall_micro']:.4f}, F1 (Micro): {metrics['f1_micro']:.4f}")
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts = texts.to(device)
                labels = labels.float().to(device)
                outputs = model(texts).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        if epoch % 2 == 0:
            print(f'Epoch: {epoch:4.0f} / {epochs} | Training Loss: {train_loss:.5f}, Validation Loss: {val_loss:.5f}')

    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            preds = model(texts).squeeze() > 0.5
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)  # Per-class
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    }

if __name__ == "__main__":
    main()