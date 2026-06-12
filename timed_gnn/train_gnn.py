import sys
# Per aggiungere al path la libreria fornita dai professori
sys.path.append('/content/HGCN/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random
import numpy as np

from OneLevelGCN import GCNModel, CustomDataset, train, evaluate, EarlyStopping

def prepare_dataloaders(dataset_path, batch_size=64):
    print(f"Caricamento blindato di {dataset_path}...")
    # Usiamo 'weights_only=False' per evitare problemi di sicurezza e compatibilità
    # e carichiamo esplicitamente su cpu per il caricamento iniziale
    raw_data = torch.load(dataset_path, map_location=torch.device('cpu'), weights_only=False)

    random.shuffle(raw_data)

    graphs = []
    labels = []

    for item in raw_data:
        g = item['graph']

        # Nella libreria si usano -1 come maschera di padding per ignorare i dati.
        # Noi avevamo mappato i pezzi neri a -1, li cambiamo in 2.0
        # per non farli cancellare dalla loro rete.
        g.x[g.x[:, 1] == -1.0, 1] = 2.0

        graphs.append(g)
        # Sottraiamo 1 al target perché PyTorch necessita classi da 0 a 4, non da 1 a 5
        labels.append(item['mate_in'] - 1)

    # Convertiamo le label in un tensore
    labels = torch.tensor(labels, dtype=torch.long)

    # Split 80% / 10% / 10%
    total = len(graphs)
    train_len = int(total * 0.8)
    val_len = int(total * 0.1)

    # Usiamo il CustomDataset creato dai professori
    train_dataset = CustomDataset(graphs[:train_len], labels[:train_len])
    val_dataset = CustomDataset(graphs[train_len:train_len+val_len], labels[train_len:train_len+val_len])
    test_dataset = CustomDataset(graphs[train_len+val_len:], labels[train_len+val_len:])

    # I DataLoader di PyTorch Geometric
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset pronto! Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader

def main():
    # --- IPERPARAMETRI (Da specifiche di progetto) ---
    BATCH_SIZE = 64
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    LAYERS = 4
    HIDDEN_DIM = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo in uso: {device}")

    train_loader, val_loader, test_loader = prepare_dataloaders("gnn_training_dataset.pt", BATCH_SIZE)

    print("Inizializzazione (OneLevelGCN)...")

    # Inizializziamo il modello dei prof passando tutte le configurazioni richieste dalla loro classe
    model = GCNModel(
        num_comb_features=3, # [tipo, colore, tempo]
        gcn_hidden_dims=[HIDDEN_DIM] * LAYERS,
        fc_hidden_dims=[128, 64],
        output_dim=5, # I 5 livelli di matto
        pooling_method='max',

        # Settaggi di regolarizzazione estratti dal loro paper
        gcn_batch_norm_flag=[True] * LAYERS,
        fc_batch_norm_flag=[True, True],
        gcn_momentum=[0.1] * LAYERS,
        fc_momentum=[0.1, 0.1],
        gcn_eps=[1e-5] * LAYERS,
        fc_eps=[1e-5, 1e-5],
        gcn_dropout_flag=[True] * LAYERS,
        fc_dropout_flag=[True, True],
        gcn_dropout_rate=[0.3] * LAYERS,
        fc_dropout_rate=[0.3, 0.3],
        gcn_activation=['relu'] * LAYERS,
        fc_activation=['relu', 'relu'],
        gcn_skip_connections=[True] * LAYERS
    ).to(device)
    model = model.to(device)

    # Setup dell'ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Usiamo il loro modulo di EarlyStopping
    early_stopping = EarlyStopping(patience=10, mode='max')
    l1_lambda = 1e-5

    print("\n INIZIO ADDESTRAMENTO ")
    best_val_acc = 0

    for epoch in range(EPOCHS):
        # Usiamo le loro esatte funzioni di train e evaluate!
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, l1_lambda)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoca {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Salviamo il modello migliore
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_timed_gnn.pth")

        # Controllo anti-overfitting
        if early_stopping(val_acc):
            print("Early stopping attivato: la rete ha smesso di migliorare.")
            break

    # TEST FINALE (Fase 4 del progetto)
    print("\n--- Valutazione Finale sul Test Set Interno ---")
    model.load_state_dict(torch.load("best_timed_gnn.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Accuratezza finale su dati mai visti: {test_acc:.4f}")
    print("Addestramento completato! Modello salvato come 'best_timed_gnn.pth'")

if __name__ == '__main__':
    main()
