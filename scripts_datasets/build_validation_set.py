import pandas as pd
import json
import os

def create_validation_set(csv_path, output_json_path):
    """
    Legge un dataset di puzzle e crea un set bilanciato di 150 problemi.
    """
    if not os.path.exists(csv_path):
        print(f"Errore: Il file {csv_path} non esiste. Scarica i puzzle e assicurati che il nome sia corretto.")
        return

    print(f"Caricamento del dataset {csv_path}...")
    df = pd.read_csv(csv_path)
    
    validation_set = []
    problemi_per_n = 30 # Quanti problemi per ogni n
    
    print("Estrazione dei problemi stratificati per profondita' di matto (n)...")
    
    for n in range(1, 6):
        theme_tag = f'mateIn{n}'
        
        # NOTA: Se il tuo CSV ha nomi di colonne diversi, modificali qui sotto
        # Ad esempio, in alcuni dataset la colonna dei temi si chiama 'Themes' o 'themes'
        colonna_temi = 'Themes' if 'Themes' in df.columns else 'themes'
        colonna_mosse = 'Moves' if 'Moves' in df.columns else 'moves'
        colonna_fen = 'FEN' if 'FEN' in df.columns else 'fen'
        
        puzzle_filtrati = df[df[colonna_temi].str.contains(theme_tag, na=False, case=False)]
        campioni = puzzle_filtrati.head(problemi_per_n)
        
        for index, row in campioni.iterrows():
            # Estrarre l'ID del puzzle in modo sicuro
            puzzle_id = row.get('PuzzleId') if 'PuzzleId' in row else row.get('PuzzleId', str(index))
            
            validation_set.append({
                "id": str(puzzle_id),
                "fen": row[colonna_fen],
                "uci_moves": row[colonna_mosse],
                "mate_in": n
            })
            
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(validation_set, f, indent=4)
        
    print(f"SUCCESSO: Validation set creato! {len(validation_set)} problemi salvati in '{output_json_path}'.")

if __name__ == "__main__":
    # Inserisci qui il nome del file CSV che hai scaricato
    input_csv = "lichess_puzzle_transformed.csv" # o il nome del file di Kaggle
    output_json = "held_out_validation.json"
    
    create_validation_set(input_csv, output_json)