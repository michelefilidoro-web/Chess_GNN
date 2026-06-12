import torch
import json
import sys
sys.path.append('/content/HGCN/src')
from OneLevelGCN import GCNModel
from OneLevelGCN import prepare_data
import chess
from torch_geometric.data import Data

def convert_fen_to_graph(fen, uci_moves):
    board = chess.Board(fen)
    time_feature = 0.0

    x = torch.zeros((64, 3), dtype=torch.float)
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            x[square][0] = piece.piece_type
            x[square][1] = 1.0 if piece.color == chess.WHITE else -1.0
            x[square][2] = time_feature

    edge_sources, edge_targets = [], []
    for move in board.legal_moves:
        edge_sources.append(move.from_square)
        edge_targets.append(move.to_square)

    for square in range(64):
        attackers = board.attackers(chess.WHITE, square) | board.attackers(chess.BLACK, square)
        for attacker in attackers:
            edge_sources.append(attacker)
            edge_targets.append(square)

    edges = list(set(zip(edge_sources, edge_targets)))
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src, dst = zip(*edges)
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('held_out_validation.json', 'r') as f:
        val_data = json.load(f)

    model = GCNModel(num_comb_features=3, gcn_hidden_dims=[128]*4, fc_hidden_dims=[128, 64],
                     output_dim=5, pooling_method='max', gcn_batch_norm_flag=[True]*4,
                     fc_batch_norm_flag=[True, True], gcn_momentum=[0.1]*4, fc_momentum=[0.1, 0.1],
                     gcn_eps=[1e-5]*4, fc_eps=[1e-5, 1e-5], gcn_dropout_flag=[True]*4,
                     fc_dropout_flag=[True, True], gcn_dropout_rate=[0.3]*4, fc_dropout_rate=[0.3, 0.3],
                     gcn_activation=['relu']*4, fc_activation=['relu', 'relu'], gcn_skip_connections=[True]*4).to(device)

    model.load_state_dict(torch.load("best_timed_gnn.pth", map_location=device))
    model.eval()

    correct = 0
    results = [] # Lista per salvare i risultati

    with torch.no_grad():
        for item in val_data:
            graph = convert_fen_to_graph(item['fen'], item['uci_moves']).to(device)
            graph.x[graph.x[:, 1] == -1.0, 1] = 2.0

            output = model(graph)
            pred = output.argmax(dim=1).item() + 1

            # 2. Salvo predizione e target
            results.append({
                'id': item.get('id', 'N/A'),
                'target': item['mate_in'],
                'pred': pred
            })

            if pred == item['mate_in']:
                correct += 1

    print(f"Accuratezza finale: {correct/len(val_data):.4f}")

    # 3. Write nel file json
    with open('final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Dati esportati con successo in 'final_results.json'!")

if __name__ == '__main__':
    test_model()
