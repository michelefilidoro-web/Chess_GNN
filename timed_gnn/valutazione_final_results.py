import json

def valuta_gnn(file_json='final_results.json'):
    print(f"Caricamento dei risultati della GNN da: {file_json}...")

    try:
        with open(file_json, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Errore: file non trovato. Assicurati di aver lanciato prima test_best_gnn.py!")
        return

    puzzle_totali = len(data)
    risposte_corrette = 0

    # Inizializziamo i contatori per ogni profondità di matto (da 1 a 5)
    statistiche_per_matto = {
        1: {"totali": 0, "corrette": 0},
        2: {"totali": 0, "corrette": 0},
        3: {"totali": 0, "corrette": 0},
        4: {"totali": 0, "corrette": 0},
        5: {"totali": 0, "corrette": 0}
    }

    # Calcoliamo le statistiche
    for item in data:
        target = item['target']
        predizione = item['pred']

        # Aggiorniamo il totale per questa profondità
        if target in statistiche_per_matto:
            statistiche_per_matto[target]["totali"] += 1

            # Se la rete ha indovinato, aggiorniamo i successi
            if predizione == target:
                risposte_corrette += 1
                statistiche_per_matto[target]["corrette"] += 1

    # --- STAMPA DEL REPORT ---
    accuratezza_totale = (risposte_corrette / puzzle_totali) * 100

    print("\n" + "="*40)
    print(" REPORT FINALE GNN (Il tuo Modello)")
    print("="*40)
    print(f"Puzzle analizzati: {puzzle_totali}")
    print(f"Risposte corrette: {risposte_corrette}")
    print(f"ACCURATEZZA GLOBALE: {accuratezza_totale:.2f}%\n")

    print("--- Dettaglio per compilare la tabella ---")
    for matto_in, stats in statistiche_per_matto.items():
        if stats["totali"] > 0:
            acc_parziale = (stats["corrette"] / stats["totali"]) * 100
            print(f"Matto in {matto_in} -> {acc_parziale:.2f}% ({stats['corrette']}/{stats['totali']})")
        else:
            print(f"Matto in {matto_in} -> N/A (0 puzzle di questo tipo)")

if __name__ == '__main__':
    valuta_gnn()
