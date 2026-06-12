import json

def calcola_accuratezza(file_json):
    """Funzione helper per leggere un JSON e calcolare l'accuratezza."""
    try:
        with open(file_json, 'r') as f:
            data = json.load(f)
            
        puzzle_totali = len(data)
        risposte_corrette = sum(1 for item in data if item['pred'] == item['target'])
        
        return (risposte_corrette / puzzle_totali) * 100
        
    except FileNotFoundError:
        print(f"Errore: File '{file_json}' non trovato!")
        return None

def valuta_ablation(file_timed='final_results.json', file_untimed='untimed_results.json'):
    print("Analisi dinamica dei risultati in corso...")
    
    # Calcolo dinamico da entrambi i file
    acc_timed = calcola_accuratezza(file_timed)
    acc_untimed = calcola_accuratezza(file_untimed)
    
    # Se uno dei due file manca, blocchiamo tutto
    if acc_timed is None or acc_untimed is None:
        return

    delta = acc_timed - acc_untimed

    print("\n" + "="*50)
    print("📉 REPORT ABLATION STUDY (Confronto Dinamico) 📉")
    print("="*50)
    print(f"File Modello Originale (Timed): '{file_timed}'")
    print(f"File Modello di Controllo (Untimed): '{file_untimed}'\n")
    
    print(f"Accuratezza Timed:   {acc_timed:.2f}%")
    print(f"Accuratezza Untimed: {acc_untimed:.2f}%")
    print("-" * 50)
    
    # Usiamo il valore assoluto per il testo, ma indichiamo se è positivo o negativo
    segno = "+" if delta >= 0 else ""
    print(f"Differenza (Delta):  {segno}{delta:.2f}%\n")
    
    if delta > 0:
        print("Conclusione: Il tempo ha un impatto POSITIVO sulle performance.")
    elif delta < 0:
        print("Conclusione: Il tempo ha un impatto NEGATIVO sulle performance.")
    else:
        print("Conclusione: Il tempo NON ha alcun impatto sulle performance.")

if __name__ == '__main__':
    # Assicurati che i nomi dei file siano quelli che hai effettivamente salvato
    valuta_ablation(file_timed='final_results.json', file_untimed='untimed_results.json')
