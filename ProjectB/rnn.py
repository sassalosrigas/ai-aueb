import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logisticregression import DatabaseLoader


def load_glove_embeddings(glove_path, embedding_dim=300):
    """Sinarthsh pou fortwnwi ta etoima embeddings mesw tou glove"""
    word2idx = {"<PAD>": 0, "<UNK>": 1}  #eidika tokens
    vectors = [np.zeros(embedding_dim), np.zeros(embedding_dim)] #antoistoixa vectors
    idx = 2  #index ksekinaei apo 2 giati 0 kai 1 einai desmeumena

    print(f"Loading GloVe embeddings from {glove_path}...")

    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]  #eksagogh lekshs kathe fora
            vector = np.asarray(values[1:], dtype='float32') #eksagwgh vector
            vectors.append(vector)
            word2idx[word] = idx
            idx += 1

            
            if i % 100000 == 0: #kathe 10000 lekseis typwse  
                print(f"Loaded {i} words...")

    embedding_matrix = np.vstack(vectors)  #stoivagma embedding
    print(f"GloVe embeddings loaded! Total words: {len(word2idx)}")

    return word2idx, embedding_matrix


# Dataset class
class TextDataset(Dataset):
    """Dataset gia katataksh keimenou"""
    def __init__(self, texts: List[str], labels: List[int], word2idx: Dict[str, int], max_length: int = 200):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        indexed = [self.word2idx.get(t, 1) for t in tokens][:self.max_length] #<UNK> an den uparxei h leksh
        indexed += [0] * (self.max_length - len(indexed))  #sequence se max length
        return torch.tensor(indexed, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)



class BiRNNClassifier(nn.Module):
    """Bidirectional classifier me LSTM kelia"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pretrained_embeddings, freeze_emb=False, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings)) #load hdh ekpaideumenwn embeddings
        if freeze_emb:
            self.embedding.weight.requires_grad = False #freeze efoson xreiastei
        self.rnn = nn.LSTM(
            input_size=embed_dim,  
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0  # dropout an parapanw apo ena layer
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  #output layer * 2 epeidh einai bidirectional

    def forward(self, x):
        emb = self.embedding(x)  #kane metatroph se embeddings
        rnn_out, _ = self.rnn(emb)  #perase mesw LSTM
        pooled, _ = torch.max(rnn_out, dim=1)  # Global max pooling
        logits = self.fc(pooled).squeeze(1)  #apply strwsh
        return logits


def build_dataset_splits():
    """Sinarthsh gia fortwma kai xwrisma dataset"""
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)

    train_texts = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    combined = list(zip(train_texts, train_labels))
    np.random.shuffle(combined)
    train_texts, train_labels = zip(*combined)

    dev_size = int(0.1 * len(train_texts)) #10% data xrhsimopoieitai gia validation
    dev_texts = train_texts[:dev_size]
    dev_labels = train_labels[:dev_size]
    final_train_texts = train_texts[dev_size:]
    final_train_labels = train_labels[dev_size:]

    test_texts = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    return list(final_train_texts), list(final_train_labels), \
           list(dev_texts), list(dev_labels), \
           list(test_texts), list(test_labels)



def train_model(model, train_loader, dev_loader, epochs=6, lr=0.001, device='cpu', gradient_clip=5.0, patience=3):
    """Sinarthsh ekpaideushs tou montelou"""
    criterion = nn.BCEWithLogitsLoss() #classification me loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    train_losses, dev_losses = [], []
    best_dev_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0  #counter gia efarmogh early stopping

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)  # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()  #validation
        dev_epoch_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                dev_epoch_loss += loss.item()
        avg_dev_loss = dev_epoch_loss / len(dev_loader)
        dev_losses.append(avg_dev_loss)
        scheduler.step(avg_dev_loss)  # Update learning rate

        print(f"Epoch {epoch} | Train Loss: {train_losses[-1]:.4f} | Dev Loss: {avg_dev_loss:.4f}")

        #kwdikas gia early stopping
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0  
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    #load ths kaluterhs katastashs
    model.load_state_dict(best_model_state)
    
    #plot diagrammatos
    plt.figure()
    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 1), dev_losses, label='Dev Loss')
    plt.legend()
    plt.show()
    
    return model

def evaluate_model(model, test_loader, device='cpu'):
    """Sinarthsh aksiologhshs meta apo thn ekpaideush"""
    model.eval()  
    all_preds, all_labels = [], []  #provlepseis/pragmatikes etiketes
    with torch.no_grad():   #apenergopoihsh parakolouthishs
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)  #metafora sth suskeuh
            logits = model(batch_x)     #provlepsh
            probs = torch.sigmoid(logits).cpu().numpy()  #sigmoid gia lipsh pithanothtwn
            preds = (probs > 0.5).astype(int) #kathgoriopoihsh me orio to 0.5 
            all_preds.extend(preds) #apothikeush provlepsewn
            all_labels.extend(batch_y.numpy())  #apothikeush labels

    #upologismos metrikwn katallhlwn gia aksiologhsh 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #orismos siskeuhs
    print(f"Using device: {device}")

    print("Loading dataset splits...")
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = build_dataset_splits()
    print(f"Dataset loaded! Training: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")

    w2v_path = "C:\\Users\\dodor\\Downloads\\glove.42B.300d\\glove.42B.300d.txt"  #path gia to arxeio pou exei ta embeddings
    print("Loading embeddings...")
    word2idx, embedding_matrix = load_glove_embeddings(w2v_path, embedding_dim=300)
    print(f"Embeddings loaded! Vocab size: {len(word2idx)}, Embedding shape: {embedding_matrix.shape}")

    #datasets ekpaideushs, aksiologhshs kai dokimhs
    train_dataset = TextDataset(train_texts, train_labels, word2idx, max_length=200)
    dev_dataset = TextDataset(dev_texts, dev_labels, word2idx, max_length=200)
    test_dataset = TextDataset(test_texts, test_labels, word2idx, max_length=200)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Initializing model...")
    model = BiRNNClassifier(
        vocab_size=embedding_matrix.shape[0], #megethos leksikou
        embed_dim=embedding_matrix.shape[1], #diastaseis embedding
        hidden_dim=256, #krimmenh diastash LSTM
        num_layers=2, # arithmos strqmatwn
        pretrained_embeddings=embedding_matrix, # ekpaideumena embeddings
        freeze_emb=True,
        dropout=0.5
    ).to(device)
    print("Model initialized!")

    print("Starting training...")
    model = train_model(model, train_loader, dev_loader, epochs=15, lr=1e-3, device=device, gradient_clip=5.0)
    print("Training completed!")

    print("Evaluating model...")
    evaluate_model(model, test_loader, device=device)
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
