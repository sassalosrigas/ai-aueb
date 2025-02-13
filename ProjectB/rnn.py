import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict
from naiveBayes import DatabaseLoader

def load_glove_embeddings(glove_path, embedding_dim=300):
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    vectors = [np.zeros(embedding_dim), np.zeros(embedding_dim)]
    idx = 2

    print(f"Loading GloVe embeddings from {glove_path}...")

    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            vectors.append(vector)
            word2idx[word] = idx
            idx += 1

            # Prints progress
            if i % 100000 == 0:
                print(f"Loaded {i} words...")

    embedding_matrix = np.vstack(vectors)
    print(f"GloVe embeddings loaded! Total words: {len(word2idx)}")

    return word2idx, embedding_matrix


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], word2idx: Dict[str, int], max_length: int = 200):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        indexed = [self.word2idx.get(t, 1) for t in tokens][:self.max_length]
        indexed += [0] * (self.max_length - len(indexed))
        return torch.tensor(indexed, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


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

def build_dataset_splits():
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)

    train_texts = train_pos + train_neg
    train_labels = [1]*len(train_pos) + [0]*len(train_neg)
    combined = list(zip(train_texts, train_labels))
    np.random.shuffle(combined)
    train_texts, train_labels = zip(*combined)

    dev_size = int(0.1 * len(train_texts))
    dev_texts = train_texts[:dev_size]
    dev_labels = train_labels[:dev_size]
    final_train_texts = train_texts[dev_size:]
    final_train_labels = train_labels[dev_size:]

    test_texts = test_pos + test_neg
    test_labels = [1]*len(test_pos) + [0]*len(test_neg)

    return list(final_train_texts), list(final_train_labels), \
           list(dev_texts), list(dev_labels), \
           list(test_texts), list(test_labels)


def train_model(model, train_loader, dev_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, dev_losses = [], []
    best_dev_loss = float('inf')
    best_model_state = None

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        dev_epoch_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                dev_epoch_loss += loss.item()
        avg_dev_loss = dev_epoch_loss / len(dev_loader)
        dev_losses.append(avg_dev_loss)

        print(f"Epoch {epoch} | Train Loss: {train_losses[-1]:.4f} | Dev Loss: {avg_dev_loss:.4f}")
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), dev_losses, label='Dev Loss')
    plt.legend()
    plt.show()
    return model

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

    p_pos = precision_score(all_labels, all_preds, pos_label=1)
    r_pos = recall_score(all_labels, all_preds, pos_label=1)
    f_pos = f1_score(all_labels, all_preds, pos_label=1)
    p_neg = precision_score(all_labels, all_preds, pos_label=0)
    r_neg = recall_score(all_labels, all_preds, pos_label=0)
    f_neg = f1_score(all_labels, all_preds, pos_label=0)
    p_micro = precision_score(all_labels, all_preds, average='micro')
    r_micro = recall_score(all_labels, all_preds, average='micro')
    f_micro = f1_score(all_labels, all_preds, average='micro')
    p_macro = precision_score(all_labels, all_preds, average='macro')
    r_macro = recall_score(all_labels, all_preds, average='macro')
    f_macro = f1_score(all_labels, all_preds, average='macro')

    print(f"POSITIVE: P={p_pos:.3f} R={r_pos:.3f} F1={f_pos:.3f}")
    print(f"NEGATIVE: P={p_neg:.3f} R={r_neg:.3f} F1={f_neg:.3f}")
    print(f"Micro-Averaged: P={p_micro:.3f} R={r_micro:.3f} F1={f_micro:.3f}")
    print(f"Macro-Averaged: P={p_macro:.3f} R={r_macro:.3f} F1={f_macro:.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset splits...")
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = build_dataset_splits()
    print(f"Dataset loaded! Training: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")

    w2v_path = "/Users/rigas/Library/CloudStorage/OneDrive-aueb.gr/Semester 5/Artificial Intelligence/ai-aueb/ProjectB/glove.6B.300d.txt"
    
    print("Loading embeddings...")
    word2idx, embedding_matrix = load_glove_embeddings(w2v_path, embedding_dim=300)
    print(f"Embeddings loaded! Vocab size: {len(word2idx)}, Embedding shape: {embedding_matrix.shape}")

    train_dataset = TextDataset(train_texts, train_labels, word2idx, max_length=200)
    dev_dataset = TextDataset(dev_texts, dev_labels, word2idx, max_length=200)
    test_dataset = TextDataset(test_texts, test_labels, word2idx, max_length=200)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Initializing model...")
    model = BiRNNClassifier(
        vocab_size=embedding_matrix.shape[0],
        embed_dim=embedding_matrix.shape[1],
        hidden_dim=128,
        num_layers=2,  # stacked (2 layers)
        pretrained_embeddings=embedding_matrix,
        freeze_emb=False
    ).to(device)
    print("Model initialized!")

    print("Starting training...")
    model = train_model(model, train_loader, dev_loader, epochs=10, lr=1e-3, device=device)
    print("Training completed!")

    print("Evaluating model...")
    evaluate_model(model, test_loader, device=device)
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
