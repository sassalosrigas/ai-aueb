import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------- DatabaseLoader (like naivebayes.py) -------------
class DatabaseLoader:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "aclImdb_v1")
    TRAINING_POSITIVE = os.path.join(DATA_DIR, "train", "pos")
    TRAINING_NEGATIVE = os.path.join(DATA_DIR, "train", "neg")
    TEST_POSITIVE = os.path.join(DATA_DIR, "test", "pos")
    TEST_NEGATIVE = os.path.join(DATA_DIR, "test", "neg")

    @staticmethod
    def load_reviews(directory: str) -> List[str]:
        reviews = []
        for root, _, files in os.walk(directory):
            for file in files:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Keep only letters and spaces, lowercased
                    text = re.sub(r'[^a-zA-Z ]', ' ', content).lower()
                    reviews.append(text)
        return reviews

# ------------- Vocabulary Building -------------
def build_vocabulary(texts: List[str], min_freq=2):
    """
    Build a vocab dict: word -> index.
    Ignores words with frequency < min_freq.
    Index 0 -> <PAD>, 1 -> <UNK>.
    """
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    # Start with special tokens
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1
    return word2idx

# ------------- TextDataset -------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], word2idx: dict, max_length=200):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        # Map tokens to indices, 1 is <UNK>
        indexed = [self.word2idx.get(w, 1) for w in tokens]
        # Pad/truncate
        indexed = indexed[:self.max_length]
        indexed += [0] * (self.max_length - len(indexed))  # 0 -> <PAD>
        label = self.labels[idx]
        return torch.tensor(indexed, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ------------- BiRNN Classifier -------------
class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # hidden_dim * 2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)         # (batch, seq_len, embed_dim)
        rnn_out, _ = self.rnn(emb)      # (batch, seq_len, hidden_dim*2)
        # Global max pooling across seq_len
        pooled, _ = torch.max(rnn_out, dim=1)  # (batch, hidden_dim*2)
        logits = self.fc(pooled).squeeze(1)    # (batch,)
        return logits

# ------------- Train Function -------------
def train_model(model, train_loader, dev_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, dev_losses = [], []
    best_dev_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_dev_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.float().to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_dev_loss += loss.item()
        avg_dev_loss = total_dev_loss / len(dev_loader)
        dev_losses.append(avg_dev_loss)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f}")
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            best_state = model.state_dict()

    # Restore best
    model.load_state_dict(best_state)
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), dev_losses, label='Dev Loss')
    plt.legend()
    plt.show()
    return model

# ------------- Evaluate Function -------------
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    # Metrics
    precision_pos = precision_score(all_labels, all_preds, pos_label=1)
    recall_pos = recall_score(all_labels, all_preds, pos_label=1)
    f1_pos = f1_score(all_labels, all_preds, pos_label=1)

    precision_neg = precision_score(all_labels, all_preds, pos_label=0)
    recall_neg = recall_score(all_labels, all_preds, pos_label=0)
    f1_neg = f1_score(all_labels, all_preds, pos_label=0)

    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')

    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    print(f"POSITIVE:  P={precision_pos:.3f}, R={recall_pos:.3f}, F1={f1_pos:.3f}")
    print(f"NEGATIVE:  P={precision_neg:.3f}, R={recall_neg:.3f}, F1={f1_neg:.3f}")
    print(f"Micro Avg: P={precision_micro:.3f}, R={recall_micro:.3f}, F1={f1_micro:.3f}")
    print(f"Macro Avg: P={precision_macro:.3f}, R={recall_macro:.3f}, F1={f1_macro:.3f}")

# ------------- main -------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load raw data
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)

    # 2. Build train set + dev set
    train_texts = train_pos + train_neg
    train_labels = [1]*len(train_pos) + [0]*len(train_neg)
    combined = list(zip(train_texts, train_labels))
    np.random.shuffle(combined)
    train_texts, train_labels = zip(*combined)

    # 10% dev split
    dev_size = int(0.1*len(train_texts))
    dev_texts = train_texts[:dev_size]
    dev_labels = train_labels[:dev_size]
    final_train_texts = train_texts[dev_size:]
    final_train_labels = train_labels[dev_size:]

    # test set
    test_texts = test_pos + test_neg
    test_labels = [1]*len(test_pos) + [0]*len(test_neg)

    # 3. Build vocabulary from training texts
    word2idx = build_vocabulary(final_train_texts, min_freq=2)

    # 4. Create PyTorch Datasets
    train_dataset = TextDataset(final_train_texts, final_train_labels, word2idx, max_length=200)
    dev_dataset   = TextDataset(dev_texts, dev_labels, word2idx, max_length=200)
    test_dataset  = TextDataset(test_texts, test_labels, word2idx, max_length=200)

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader   = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 6. Define and train model
    vocab_size = len(word2idx)
    embed_dim = 128
    hidden_dim = 128
    num_layers = 1

    model = BiRNNClassifier(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    model = train_model(model, train_loader, dev_loader, epochs=10, lr=1e-3, device=device)

    # 7. Evaluate
    evaluate_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()
