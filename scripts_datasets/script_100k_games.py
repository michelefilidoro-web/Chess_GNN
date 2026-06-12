import chess.pgn

# Inserisci qui il nome del file che hai scaricato ed estratto
input_pgn_file = "elite_db.pgn"
# Il nome del nuovo file che conterrà le tue 100k partite
output_pgn_file = "lichess_elite_100k.pgn"

limite_partite = 100000
partite_estratte = 0

print(f"Inizio estrazione di {limite_partite} partite...")

# Apriamo il file originale in lettura e il nuovo in scrittura
with open(input_pgn_file, "r", encoding="utf-8") as in_file, \
     open(output_pgn_file, "w", encoding="utf-8") as out_file:
    
    while partite_estratte < limite_partite:
        # Legge una singola partita dal file PGN
        game = chess.pgn.read_game(in_file)
        
        # Se il file finisce prima del limite, esce dal ciclo in sicurezza
        if game is None:
            print("Fine del file raggiunta prima del limite.")
            break
        
        # Scrive la partita estratta nel nuovo file
        # L'oggetto exporter si occupa di formattare correttamente l'output in PGN
        exporter = chess.pgn.FileExporter(out_file)
        game.accept(exporter)
        
        partite_estratte += 1
        
        # Un piccolo print di aggiornamento ogni 10k partite
        if partite_estratte % 10000 == 0:
            print(f"Estratte {partite_estratte} partite...")

print(f"Operazione completata! Trovate e salvate {partite_estratte} partite in '{output_pgn_file}'.")