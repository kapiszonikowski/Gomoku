# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from Gomoku import Gomoku
from Gomoku_NN import GomokuNet


@dataclass
class GameSnapshot:
    """Minimalny zrzut stanu wewnętrznego Gomoku potrzebny do czystych symulacji."""
    to_play: int
    swap_phase: int
    swap_substep: int
    move_count: int
    playerA_color: Optional[int]
    playerB_color: Optional[int]


def snapshot_from_game(game: Gomoku) -> GameSnapshot:
    return GameSnapshot(
        to_play=game.to_play,
        swap_phase=game.swap_phase,
        swap_substep=game.swap_substep,
        move_count=game.move_count,
        playerA_color=game.playerA_color,
        playerB_color=game.playerB_color,
    )


def apply_snapshot(game: Gomoku, snap: GameSnapshot) -> None:
    game.to_play = snap.to_play
    game.swap_phase = snap.swap_phase
    game.swap_substep = snap.swap_substep
    game.move_count = snap.move_count
    game.playerA_color = snap.playerA_color
    game.playerB_color = snap.playerB_color


class PureStepper:
    """
    „Czysty” krokomierz dla Gomoku – wykonuje akcję na kopii logiki,
    nie dotykając żywego obiektu w Twojej pętli.
    """
    def __init__(self, template_game: Gomoku):
        self._cfg = dict(
            size=template_game.row_count,
            win_length=template_game.win_length,
            exact_five=template_game.exact_five,
            use_swap2=template_game.use_swap2,
        )

    def valid_moves(self, state: np.ndarray, snap: GameSnapshot) -> np.ndarray:
        g = Gomoku(**self._cfg)
        apply_snapshot(g, snap)
        return g.get_valid_moves(state)

    def step(self, state: np.ndarray, snap: GameSnapshot, action: int) -> Tuple[np.ndarray, GameSnapshot]:
        g = Gomoku(**self._cfg)
        apply_snapshot(g, snap)
        new_state = state.copy()
        new_state = g.get_next_state(new_state, action, g.to_play)  # player arg ignorowany w Gomoku
        new_snap = snapshot_from_game(g)
        return new_state, new_snap

    def value_and_terminated(self, state: np.ndarray, snap: GameSnapshot, last_action: Optional[int]) -> Tuple[int, bool]:
        # Używamy tej samej definicji co w Gomoku (wartość z perspektywy ostatniego ruchu).
        g = Gomoku(**self._cfg)
        apply_snapshot(g, snap)
        if last_action is None:
            # Metoda wykrywa terminal także bez ostatniej akcji
            has_moves = np.any(self.valid_moves(state, snap))
            return 0, not has_moves
        value, term = g.get_value_and_terminated(state, last_action)
        return value, term


class Node:
    def __init__(self, stepper: PureStepper, state: np.ndarray, snap: GameSnapshot,
                 parent: Optional['Node']=None, action_taken: Optional[int]=None,
                 NN: bool=False, Test: bool=False,
                 net: Optional[GomokuNet]=None, device=None):               # <-- DODANE
        self.net = net                                                     # <-- DODANE
        self.device = device                                               # <-- DODANE
        self.stepper = stepper
        self.state = state
        self.snap = snap
        self.parent = parent
        self.action_taken = action_taken
        self.NN = NN
        self.Test = Test

        self.children: List['Node'] = []
        self.visit_count = 0
        self.value_sum = 0.0

        # legalne akcje z pozycji (uwzględnia fazy swap2)
        self.expandable_moves = self.stepper.valid_moves(self.state, self.snap).astype(np.uint8)

    def is_fully_expanded(self) -> bool:
        return np.sum(self.expandable_moves) == 0

    def select(self, c_ucb: float) -> 'Node':
        """Wybierz dziecko wg UCB1."""
        best_child, best_ucb = None, -np.inf
        for child in self.children:
            u = self._ucb(child, c_ucb)
            if u > best_ucb:
                best_ucb, best_child = u, child
        return best_child

    def _q(self) -> float:
        return 0.0 if self.visit_count == 0 else (self.value_sum / self.visit_count)

    def _ucb(self, child: 'Node', c_ucb: float) -> float:
        if child.visit_count == 0:
            return float('inf')
        q = child.value_sum / child.visit_count  # w [-1, 1]
        return q + c_ucb * math.sqrt(math.log(max(1, self.visit_count)) / child.visit_count)

    def expand(self) -> 'Node':
        legal_idx = np.where(self.expandable_moves == 1)[0]

        if self.NN:
            if self.net is None:
                raise RuntimeError("NN=True, ale nie przekazano instancji sieci (net=None).")

            # forward przez sieć; nasza sieć akceptuje numpy (po recent patchu)
            with torch.no_grad():
                # jeśli masz GPU i chcesz wymusić tensory na device:
                # sieć sama zamieni numpy->tensor na CPU; bezpiecznie przerzucić na device wyjścia
                out = self.net(self.state)  # dict: {'policy_logits': (1,A), 'value': (1,)}
                logits_t = out['policy_logits']  # tensor (1, A)
                # na CPU do numpy
                logits = logits_t.squeeze(0).detach().cpu().numpy()

            # maskowanie ruchów nielegalnych
            masked_logits = np.full_like(logits, -np.inf, dtype=np.float32)
            masked_logits[legal_idx] = logits[legal_idx]

            # wybór akcji (argmax logitów zamaskowanych)
            action = int(np.argmax(masked_logits))

            if self.Test:
                print(f"NN action: {action}  logit={masked_logits[action]:.3f}")
        else:
            action = int(np.random.choice(legal_idx))

        # zabezpieczenie: gdyby jednak nie było ruchów (terminal)
        if len(legal_idx) == 0:
            # nic do rozwinięcia — zwróć siebie (lub możesz podnieść wyjątek)
            return self

        self.expandable_moves[action] = 0
        child_state, child_snap = self.stepper.step(self.state, self.snap, action)
        child = Node(self.stepper, child_state, child_snap,
                    parent=self, action_taken=action,
                    NN=self.NN, Test=self.Test, net=self.net, device=self.device)   # <-- DODANE
        self.children.append(child)
        return child


    def simulate(self) -> float:
        """
        Prosty rollout losowy aż do terminala.
        Zwraca wartość z PERSPEKTYWY gracza będącego na ruchu w tym węźle (self.snap.to_play).
        """
        # Jeśli węzeł powstał po ruchu parenta, sprawdź terminal:
        val, term = self.stepper.value_and_terminated(self.state, self.snap, self.action_taken)
        # val jest z perspektywy ostatniego ruchu; my chcemy perspektywę aktualnego na ruchu:
        if term:
            if val == 0:
                return 0.0
            # Ostatni ruch wykonał przeciwnik aktualnego self.snap.to_play -> to dla nas porażka:
            return -1.0

        rollout_state = self.state.copy()
        rollout_snap = GameSnapshot(**vars(self.snap))  # płytka kopia dataclass
        # Znak +1 oznacza „ostatni ruch zrobił gracz startowy w tym węźle”
        start_to_play = rollout_snap.to_play

        last_action = None
        while True:
            valid = self.stepper.valid_moves(rollout_state, rollout_snap)
            moves = np.where(valid == 1)[0]
            action = int(np.random.choice(moves))
            rollout_state, rollout_snap = self.stepper.step(rollout_state, rollout_snap, action)
            last_action = action

            val, term = self.stepper.value_and_terminated(rollout_state, rollout_snap, last_action)
            if term:
                if val == 0:
                    return 0.0
                # Jeśli zwyciężył ostatni ruch, sprawdzamy czy ten ruch należał do startowego gracza w tym węźle:
                # startowy to start_to_play; po nieparzystej liczbie posunięć wygrał startowy.
                # Tu prościej: jeśli po ruchu, który zakończył grę, 'to_play' w snapie wskazuje PRZECIWNIKA
                # bieżącego gracza w tym węźle, to znaczy że wygrał gracz, który właśnie zagrał.
                # Z perspektywy gracza startowego (self.snap.to_play): zwycięstwo = +1, porażka = -1.
                # Ponieważ self.stepper.value_and_terminated ocenia z perspektywy „ostatniego ruchu”,
                # dla nas to zawsze +1 jeśli ostatni ruch należał do gracza startowego.
                # Aby nie śledzić parzystości, porównamy: kto wykonał ostatni ruch?
                # Ostatni ruch wykonał przeciwnik aktualnego rollout_snap.to_play.
                last_mover = -rollout_snap.to_play
                return 1.0 if last_mover == start_to_play else -1.0

    def backpropagate(self, value: float) -> None:
        """Propagacja naprzemienna znaków w górę drzewa."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, game: Gomoku, C: float = 1.41, num_searches: int = 800,
                 NN: bool = False, Test: bool = False,
                 net: Optional[GomokuNet] = None, device=None):          # <-- DODANE
        self.game = game
        self.C = float(C)
        self.num_searches = int(num_searches)
        self.stepper = PureStepper(template_game=game)
        self.NN = NN
        self.Test = Test
        self.net = net                                                 # <-- DODANE
        self.device = device                                           # <-- DODANE


        # do raportowania po ostatnim wyszukiwaniu:
        self.last_visit_counts: Optional[np.ndarray] = None
        self.last_children: List[Tuple[int, int]] = []  # (action, visits)

    def _root(self, state: np.ndarray) -> Node:
        snap = snapshot_from_game(self.game)
        return Node(self.stepper, state.copy(), snap,
                    parent=None, action_taken=None,
                    NN=self.NN, Test=self.Test,
                    net=self.net, device=self.device)     # <-- DODANE

    def search(self, state: np.ndarray) -> np.ndarray:
        root = self._root(state)

        for _ in range(self.num_searches):
            node = root
            # SELEKCJA
            while node.is_fully_expanded():
                node = node.select(self.C)

            # EKSPANSJA + SYMULACJA
            # Jeśli obecny węzeł nie jest terminalny – rozwiń i playout z dziecka
            val, term = self.stepper.value_and_terminated(node.state, node.snap, node.action_taken)
            if term:
                # val: +1 z perspektywy ostatniego ruchu; z perspektywy gracza w node – odwrotnie
                value_for_node_player = 0.0 if val == 0 else -1.0
            else:
                child = node.expand()
                value_for_node_player = child.simulate()

            # BACKUP
            node.backpropagate(value_for_node_player)

        # Rozkład wizyt w korzeniu -> polityka bota
        action_probs = np.zeros(self.game.action_size, dtype=np.float32)
        for ch in root.children:
            action_probs[ch.action_taken] = ch.visit_count
        s = np.sum(action_probs)
        if s > 0:
            action_probs /= s

        # zapamiętaj do inspekcji
        self.last_visit_counts = action_probs * s  # faktyczne wizyty
        self.last_children = [(ch.action_taken, ch.visit_count) for ch in root.children]
        return action_probs

    # —————— Narzędzia do podglądu/raportu ——————
    def visit_distribution(self, topk: int = 10) -> List[Dict]:
        """
        Zwraca listę „TOP k” ruchów po ostatnim wyszukiwaniu:
        dict(action=..., kind=..., rc=..., visits=..., prob=...).
        """
        if self.last_visit_counts is None:
            return []
        counts = self.last_visit_counts
        probs = counts / max(1.0, np.sum(counts))
        idx = np.argsort(counts)[::-1]
        out = []
        k = min(topk, np.sum(counts > 0).astype(int))
        for a in idx[:k]:
            v = counts[a]
            if v <= 0:
                break
            if a < self.game.board_actions:
                r = a // self.game.column_count
                c = a % self.game.column_count
                kind = "BOARD"
                rc = f"{r+1} {c+1}"
            else:
                # etykiety specjalnych przycisków Swap2
                if a == self.game.SWAP_CHOOSE_WHITE:
                    kind, rc = "SWAP2", "CHOOSE_WHITE"
                elif a == self.game.SWAP_CHOOSE_BLACK:
                    kind, rc = "SWAP2", "CHOOSE_BLACK"
                elif a == self.game.SWAP_PLACE_TWO:
                    kind, rc = "SWAP2", "PLACE_TWO"
                else:
                    kind, rc = "SPECIAL", str(a)
            out.append(dict(action=int(a), kind=kind, rc=rc, visits=int(v), prob=float(probs[a])))
        return out
