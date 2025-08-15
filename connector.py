from Gomoku import Gomoku
from MCTS import MCTS
from GUMBLE import GumbelMCTS
import numpy as np
from Gomoku_NN import GomokuNet  # <-- DODAJ
import torch                     # <-- DODAJ

if __name__ == "__main__":
    size = 8
    gomoku = Gomoku(size=size, win_length=5, exact_five=False, use_swap2=True)
    state = gomoku.get_initial_state()

    # --- NN: utwórz i ustaw urządzenie ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GomokuNet(board_size=size).to(device).eval()

    gmcts = GumbelMCTS(gomoku, num_simulations=1000, num_candidates=None,
                       cvisit=50, cscale=1, keep_ratio=0.5, root_dirichlet_epsilon=1)

    # Przekazujemy sieć do MCTS (NN=True uruchomi ścieżkę NN)
    mcts = MCTS(gomoku, C=1.41, num_searches=500, NN=True, Test=False, net=net, device=device)

    # Zagraj vs. MCTS (człowiek jako „A”, bot jako „B” – to naturalnie rozstrzygnie się w trakcie Swap2)
    human_label = "B"
    while True:
        print(state)
        who = gomoku.current_player_label()
        print(f"to_play {who} (1=Black, -1=White): {gomoku.to_play}")

        valid = gomoku.get_valid_moves(state)

        if who == human_label:
            # Człowiek
            if gomoku.use_swap2 and gomoku.swap_phase in (2, 4):
                msg = f"Swap2 – wybierz: {gomoku.SWAP_CHOOSE_WHITE}=CHOOSE_WHITE, {gomoku.SWAP_CHOOSE_BLACK}=CHOOSE_BLACK"
                if gomoku.swap_phase == 2:
                    msg += f", {gomoku.SWAP_PLACE_TWO}=PLACE_TWO"
                print(msg)

            raw = input(f"Ruch (indeks 0..{gomoku.action_size-1} lub 'r c'): ").strip()
            if " " in raw:
                r_s, c_s = raw.split()
                r, c = int(r_s)-1, int(c_s)-1
                action = r * gomoku.column_count + c
            else:
                action = int(raw)

            if action < 0 or action >= gomoku.action_size or valid[action] == 0:
                print("Ruch niedozwolony.")
                continue
        else:
            # BOT – MCTS
            probs = mcts.search(state)
            action = int(np.argmax(probs))

        # Wykonaj ruch w prawdziwej grze
        state = gomoku.get_next_state(state, action, gomoku.to_play)
        value, terminal = gomoku.get_value_and_terminated(state, action)

        if who != human_label:
            # Raport: rozkład wizyt po ruchu bota
            print("MCTS: TOP wizyty po ostatnim wyszukiwaniu:")
            for d in mcts.visit_distribution(topk=10):
                print(f"  a={d['action']:>3} [{d['kind']:<6}] {d['rc']:<10}  visits={d['visits']:<5}  p={d['prob']:.3f}")

        if terminal:
            print(state)
            if value == 1:
                winner_color = -gomoku.to_play
                winner_label = "A" if winner_color == gomoku.playerA_color else "B"
                print(f"Wygrał gracz {winner_label} (kolor {winner_color}).")
            else:
                print("Remis")
            break
