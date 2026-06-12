import json
import re

def analizza_risultati_llm(file_json):
    # 1. Carichiamo i dati
    print(f"Caricamento del file: {file_json}...")
    with open(file_json, 'r') as f:
        data = json.load(f)

    puzzle_totali = len(data)
    risposte_corrette = 0
    
    # Dizionario per salvare le statistiche separate per profondità (da 1 a 5)
    statistiche_per_matto = {
        1: {"totali": 0, "corrette": 0},
        2: {"totali": 0, "corrette": 0},
        3: {"totali": 0, "corrette": 0},
        4: {"totali": 0, "corrette": 0},
        5: {"totali": 0, "corrette": 0}
    }

    print("\n--- ESTRAZIONE E CONFRONTO ---")
    
    for item in data:
        target = item['target']
        risposta_grezza = item['risposta_llm']
        
        # --- LA MAGIA DELLE REGEX ---
        # r'\d+' significa: "cerca una sequenza di 1 o più cifre numeriche"
        # re.search si ferma al PRIMO numero che trova nella stringa.
        match = re.search(r'\d+', risposta_grezza)
        
        if match:
            # Estraiamo il testo trovato e lo convertiamo in numero intero
            predizione = int(match.group())
        else:
            # Se l'LLM ha sbarellato e non ha scritto nessun numero (es. "Matto subito")
            # assegniamo -1 per indicare un errore di formattazione
            predizione = -1
            
        # Aggiorniamo i contatori totali per questa specifica profondità
        statistiche_per_matto[target]["totali"] += 1
        
        # Controllo correttezza
        if predizione == target:
            risposte_corrette += 1
            statistiche_per_matto[target]["corrette"] += 1

    # --- STAMPA DEL REPORT FINALE ---
    accuratezza_totale = (risposte_corrette / puzzle_totali) * 100
    
    print("\n" + "="*40)
    print("🏆 REPORT FINALE LLM (OLLAMA) 🏆")
    print("="*40)
    print(f"Puzzle analizzati: {puzzle_totali}")
    print(f"Risposte corrette: {risposte_corrette}")
    print(f"ACCURATEZZA GLOBALE: {accuratezza_totale:.2f}%\n")
    
    print("--- Dettaglio per profondità di matto ---")
    for matto_in, stats in statistiche_per_matto.items():
        if stats["totali"] > 0:
            acc_parziale = (stats["corrette"] / stats["totali"]) * 100
            print(f"Matto in {matto_in}: {acc_parziale:.2f}% ({stats['corrette']}/{stats['totali']})")

if __name__ == '__main__':
    # Assicurati che il nome del file corrisponda a quello che hai caricato
    analizza_risultati_llm('llm_results_ollama.json')