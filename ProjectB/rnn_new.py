import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------- DatabaseLoader (from naivebayes.py) -------------
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
                    reviews.append(re.sub(r'[^a-zA-Z ]', ' ', content).lower())
        return reviews

# ------------- TextDataset -------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], word2idx: dict, max_length: int = 200):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        indexed = [self.word2idx.get(t, 1) for t in tokens]  # 1 -> <UNK>
        indexed = indexed[:self.max_length]
        indexed += [0] * (self.max_length - len(indexed))  # 0 -> <PAD>
        return torch.tensor(indexed, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# ------------- BiRNN Classifier -------------
class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pretrained_embeddings, freeze_emb=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        if freeze_emb:
            self.embedding.weight.requires_grad = False

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        rnn_out, _ = self.rnn(emb)
        pooled, _ = torch.max(rnn_out, dim=1)
        logits = self.fc(pooled).squeeze(1)
        return logits

# ------------- Utility: Load GloVe -------------
def load_glove_embeddings(glove_path, embedding_dim=100):
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    vectors = [
        np.zeros(embedding_dim),
        np.zeros(embedding_dim)
    ]
    with open(glove_path, 'r', encoding="utf-8") as f:
        idx = 2
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word2idx[word] = idx
            vectors.append(vector)
            idx += 1
    embedding_matrix = np.vstack(vectors)
    return word2idx, embedding_matrix

# ------------- Build Data Splits -------------
def build_dataset_splits():
    train_pos_reviews = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    train_neg_reviews = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    test_pos_reviews = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    test_neg_reviews = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)

    train_texts = train_pos_reviews + train_neg_reviews
    train_labels = [1]*len(train_pos_reviews) + [0]*len(train_neg_reviews)

    combined = list(zip(train_texts, train_labels))
    np.random.shuffle(combined)
    train_texts, train_labels = zip(*combined)

    dev_size = int(0.1 * len(train_texts))
    dev_texts = train_texts[:dev_size]
    dev_labels = train_labels[:dev_size]
    final_train_texts = train_texts[dev_size:]
    final_train_labels = train_labels[dev_size:]

    test_texts = test_pos_reviews + test_neg_reviews
    test_labels = [1]*len(test_pos_reviews) + [0]*len(test_neg_reviews)

    return list(final_train_texts), list(final_train_labels), \
           list(dev_texts), list(dev_labels), \
           list(test_texts), list(test_labels)

# ------------- Training Function -------------
def train_model(model, train_loader, dev_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    dev_losses = []
    best_dev_loss = float("inf")
    best_model_state = None

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        dev_epoch_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.float().to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                dev_epoch_loss += loss.item()
        avg_dev_loss = dev_epoch_loss / len(dev_loader)
        dev_losses.append(avg_dev_loss)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f}")

        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)

    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), dev_losses, label="Dev Loss")
    plt.legend()
    plt.show()

    return model

# ------------- Evaluation Function -------------
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

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

    print(f"POSITIVE: P={precision_pos:.3f} R={recall_pos:.3f} F1={f1_pos:.3f}")
    print(f"NEGATIVE: P={precision_neg:.3f} R={recall_neg:.3f} F1={f1_neg:.3f}")
    print(f"Micro-Averaged: P={precision_micro:.3f} R={recall_micro:.3f} F1={f1_micro:.3f}")
    print(f"Macro-Averaged: P={precision_macro:.3f} R={recall_macro:.3f} F1={f1_macro:.3f}")

# ------------- main -------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = build_dataset_splits()

    # Example GloVe path
    glove_path = os.path.join("path_to_glove", "glove.6B.100d.txt")
    word2idx, glove_matrix = load_glove_embeddings(glove_path, embedding_dim=100)
    vocab_size = glove_matrix.shape[0]
    embed_dim = glove_matrix.shape[1]

    train_dataset = TextDataset(train_texts, train_labels, word2idx, max_length=200)
    dev_dataset = TextDataset(dev_texts, dev_labels, word2idx, max_length=200)
    test_dataset = TextDataset(test_texts, test_labels, word2idx, max_length=200)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BiRNNClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=1,
        pretrained_embeddings=glove_matrix,
        freeze_emb=False
    ).to(device)

    model = train_model(model, train_loader, dev_loader, epochs=10, lr=1e-3, device=device)
    evaluate_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()
