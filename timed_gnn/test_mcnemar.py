import json
import re
from statsmodels.stats.contingency_tables import mcnemar

def calcola_mcnemar():
    # 1. Carichiamo i due file
    try:
        with open('final_results.json', 'r') as f:
            gnn_data = json.load(f)
        with open('llm_results_ollama.json', 'r') as f:
            llm_data = json.load(f)
    except FileNotFoundError:
        print("Errore: Assicurati di avere entrambi i file JSON nella cartella!")
        return

    # Allineiamo i risultati usando l'ID del puzzle per essere sicuri al 100%
    gnn_dict = {item['id']: item for item in gnn_data if 'id' in item}

    # Costruiamo la tabella di contingenza

    b_gnn_vince = 0
    c_llm_vince = 0

    for llm_item in llm_data:
        p_id = llm_item['id']
        target = llm_item['target']

        # Estraiamo la predizione dell'LLM con la regex
        match = re.search(r'\d+', llm_item['risposta_llm'])
        llm_pred = int(match.group()) if match else -1
        llm_corretto = (llm_pred == target)

        # Recuperiamo la predizione della GNN per lo stesso puzzle
        if p_id in gnn_dict:
            gnn_corretto = (gnn_dict[p_id]['pred'] == target)

            # Contiamo le discordanze
            if gnn_corretto and not llm_corretto:
                b_gnn_vince += 1
            elif not gnn_corretto and llm_corretto:
                c_llm_vince += 1

    print("="*50)
    print("TEST STATISTICO DI McNEMAR ")
    print("="*50)
    print(f"Puzzle vinti dalla GNN (GNN ok, LLM ko): {b_gnn_vince}")
    print(f"Puzzle vinti dall'LLM (LLM ok, GNN ko): {c_llm_vince}")

    # Eseguiamo il test
    # Usiamo exact=True perché abbiamo un campione relativamente piccolo (150)
    table = [[0, b_gnn_vince], [c_llm_vince, 0]]
    result = mcnemar(table, exact=True)

    print(f"\n P-value calcolato: {result.pvalue:.5f}")

    # Interpretazione
    alpha = 0.05
    if result.pvalue < alpha:
        print("\nCONCLUSIONE: La differenza è STATISTICAMENTE SIGNIFICATIVA (p < 0.05).")
        print("Puoi scrivere nel report che la GNN è matematicamente superiore all'LLM su questo set.")
    else:
        print("\nCONCLUSIONE: La differenza NON è statisticamente significativa (p >= 0.05).")
        print("La differenza potrebbe essere dovuta al caso.")

if __name__ == '__main__':
    calcola_mcnemar()
