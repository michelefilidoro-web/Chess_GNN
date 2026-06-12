import torch

def crea_dataset_untimed():
    print("Caricamento del dataset originale...")
    # AGGIUNTO weights_only=False PER SUPERARE IL BLOCCO DI PYTORCH 2.6
    dataset = torch.load("gnn_training_dataset.pt", weights_only=False)

    print("Azzeramento delle feature temporali...")
    for item in dataset:
        # La feature del tempo è l'indice 2 (terza colonna) del tensore x
        item['graph'].x[:, 2] = 0.0

    print("Salvataggio del nuovo dataset...")
    torch.save(dataset, "gnn_untimed_dataset.pt")
    print("Fatto! File 'gnn_untimed_dataset.pt' creato con successo.")

if __name__ == '__main__':
    crea_dataset_untimed()
