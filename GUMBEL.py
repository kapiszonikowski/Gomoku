from __future__ import annotations

# -*- coding: utf-8 -*-
import math
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch

from Gomoku import Gomoku
from Gomoku_NN import GomokuNet
from MCTS import GameSnapshot, PureStepper, snapshot_from_game


class GumbelMCTS:
    """
    Gumbel-MCTS kompatybilny z klasą Gomoku (w tym Swap2).

    W korzeniu:
      - losowanie kandydatów Gumbel-Top-k (bez zwracania),
      - Sequential Halving (połowienie wg Q̂),
      - finalny wybór wg: score = logits + gumbel + (cvisit + maxN)*cscale*Q̂.

    Głębiej w drzewo nie schodzimy (jak w prostym wariancie GMZ dla gier planszowych);
    wartości Q̂ mierzymy rolloutem z dziecka korzenia (z poprawnym sprawdzeniem terminala
    po jego własnym `last_action`).

    Dodatki praktyczne:
      - instant win check (szukamy wygranej w 1),
      - opponent mate-in-1 (wymuszony blok),
      - temperatura i Dirichlet na korzeniu,
      - regulowane "keep_ratio" w halvingu.

    Zwraca rozkład π(a) ∝ N(a) z wizyt w korzeniu (znormalizowany).
    """

    def __init__(
        self,
        game: Gomoku,
        num_simulations: int = 384,
        num_candidates: Optional[int] = None,
        keep_ratio: float = 0.5,         # 0.5 = klasyczne "połowienie"; 0.67 trzyma dłużej więcej kandydatów
        c_ucb: float = 1.41,             # zostawione na przyszłość (wariant z UCB w węzłach głębiej)
        cvisit: float = 50.0,
        cscale: float = 1.0,
        root_dirichlet_epsilon: Optional[float] = None,  # np. 0.25
        root_dirichlet_alpha: float = 0,               # np. 0.03..0.3 (zależnie od gry/rozmiaru)
        root_logits_temperature: float = 1.0,            # T>1 spłaszcza logity, T<1 wyostrza
        tactical_check: bool = False,                     # sprawdzaj mata w 1 dla przeciwnika
        instant_win_check: bool = False,                  # sprawdzaj własnego mata w 1
        rng: Optional[np.random.Generator] = None,
        NN: bool = False,
        Test: bool = False,
        net: Optional[GomokuNet] = None,
        device=None,
    ):
        self.game = game
        self.stepper = PureStepper(template_game=game)

        self.n = int(num_simulations)
        self.m = None if num_candidates is None else int(num_candidates)
        self.keep_ratio = float(keep_ratio)
        self.c_ucb = float(c_ucb)
        self.cvisit = float(cvisit)
        self.cscale = float(cscale)
        self.root_dirichlet_epsilon = root_dirichlet_epsilon
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.root_logits_temperature = float(root_logits_temperature)
        self.tactical_check = bool(tactical_check)
        self.instant_win_check = bool(instant_win_check)

        self.rng = rng if rng is not None else np.random.default_rng()

        self.NN = NN
        self.Test = Test
        self.net = net
        self.device = device

        # raport po wyszukiwaniu:
        self.last_visit_counts: Optional[np.ndarray] = None
        self.last_children: List[Tuple[int, int]] = []

    # ========= Public API =========

    def search(self, state: np.ndarray, policy_logits: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Zwraca wektor prawdopodobieństw akcji (π) oparty o wizyty w korzeniu.
        """
        root_state = state.copy()
        root_snap = snapshot_from_game(self.game)

        if policy_logits is None and self.NN:
            if self.net is None:
                raise RuntimeError("NN=True, ale nie przekazano instancji sieci (net=None).")
            with torch.no_grad():
                out = self.net(state)
                logits_t = out['policy_logits']
                policy_logits = logits_t.squeeze(0).detach().cpu().numpy()
            if self.Test:
                print("NN policy logits obliczone.")

        valid = self.stepper.valid_moves(root_state, root_snap)
        legal_idx = np.where(valid == 1)[0]
        if legal_idx.size == 0:
            # terminal – brak ruchów
            self.last_visit_counts = np.zeros(self.game.action_size, dtype=np.float32)
            return self.last_visit_counts

        # --- szybkie taktyki (opcjonalne) ---
        if self.instant_win_check:
            win_now = self._find_my_winning_moves_now(root_state, root_snap)
            if len(win_now) > 0:
                a = win_now[0]
                probs = np.zeros(self.game.action_size, dtype=np.float32); probs[a] = 1.0
                self.last_visit_counts = probs
                return probs

        if self.tactical_check:
            forced_blocks = self._find_opponent_winning_moves_next(root_state, root_snap)
            if len(forced_blocks) == 1:
                a = forced_blocks[0]  # jedyny ruch broniący
                probs = np.zeros(self.game.action_size, dtype=np.float32); probs[a] = 1.0
                self.last_visit_counts = probs
                return probs

        # --- Priory (logity) ---
        logits = self._make_root_logits(valid, policy_logits)

        # --- Gumbel-Top-k bez zwracania: wybierz m kandydatów ---
        m = self._choose_m(legal_idx.size)
        gumbels = self.rng.gumbel(loc=0.0, scale=1.0, size=self.game.action_size).astype(np.float32)

        keys = logits + gumbels  # nielegalne mają -inf, nie wejdą do top-k
        cand = np.argsort(keys)[::-1][:m]
        cand = np.array([a for a in cand if valid[a] == 1], dtype=int)
        if cand.size == 0:
            cand = legal_idx[: min(m, legal_idx.size)]

        # --- Utwórz dzieci korzenia ---
        children: Dict[int, GumbelMCTS._Node] = {
            int(a): self._make_child(root_state, root_snap, int(a)) for a in cand
        }

        # --- Sequential Halving ---
        active = cand.copy()
        phases = max(1, int(np.ceil(np.log(max(1.0, active.size)) / np.log(max(1.25, 1.0 / max(1e-9, self.keep_ratio))))))
        # powyższa formuła daje ≥1 fazę i sensowną liczbę rund dla różnych keep_ratio

        budget_used = 0
        while phases > 0 and active.size > 0 and budget_used < self.n:
            phases -= 1
            # ile symulacji na aktywnego kandydata w tej fazie?
            per = max(1, (self.n - budget_used) // max(1, active.size * (1 + phases)))
            for a in active:
                node = children[int(a)]
                for _ in range(per):
                    v = self._simulate_from_node(node)
                    self._backup_from_node(node, v)
                budget_used += per
                if budget_used >= self.n:
                    break
            if budget_used >= self.n or active.size == 1:
                break
            # zostaw top-k wg Q̂
            q_list = [(a, self._q(children[int(a)])) for a in active]
            q_list.sort(key=lambda t: t[1], reverse=True)
            keep = max(1, int(np.ceil(len(q_list) * self.keep_ratio)))
            active = np.array([a for a, _ in q_list[:keep]], dtype=int)

        # rozdziel resztę budżetu równomiernie po aktywnych (łagodniej niż "tylko w lidera")
        while budget_used < self.n and active.size > 0:
            for a in active:
                if budget_used >= self.n:
                    break
                node = children[int(a)]
                v = self._simulate_from_node(node)
                self._backup_from_node(node, v)
                budget_used += 1

        # --- Finalny scoring (jak w Gumbel MuZero Eq. 8) i polityka z wizyt ---
        maxN = 0 if len(children) == 0 else max(ch.visit_count for ch in children.values())

        def final_score(a: int) -> float:
            ch = children[int(a)]
            q = self._q(ch)  # [-1, 1]
            return float(logits[a] + gumbels[a] + (self.cvisit + maxN) * self.cscale * q)

        # (możesz użyć 'chosen = max(children.keys(), key=final_score)' gdy chcesz deterministycznie)

        visit_counts = np.zeros(self.game.action_size, dtype=np.float32)
        for a, ch in children.items():
            visit_counts[a] = ch.visit_count
        total = float(visit_counts.sum())
        probs = visit_counts / total if total > 0 else visit_counts

        self.last_visit_counts = visit_counts
        self.last_children = sorted(
            [(int(a), int(ch.visit_count)) for a, ch in children.items()],
            key=lambda t: t[1], reverse=True,
        )
        return probs

    def visit_distribution(self, topk: int = 10) -> List[Dict]:
        """Zwraca listę TOP-k ruchów (akcja/etykieta/visits/p)."""
        if self.last_visit_counts is None:
            return []
        counts = self.last_visit_counts
        s = counts.sum()
        probs = counts / s if s > 0 else counts
        idx = np.argsort(counts)[::-1]
        out = []
        for a in idx[:topk]:
            v = int(counts[a])
            if v <= 0:
                break
            kind, rc = self._label_action(a)
            out.append(dict(action=int(a), kind=kind, rc=rc, visits=v, prob=float(probs[a])))
        return out

    # ========= Wnętrzności =========

    class _Node:
        __slots__ = ("state", "snap", "visit_count", "value_sum", "expandable", "last_action")

        def __init__(self, state: np.ndarray, snap: GameSnapshot, stepper: PureStepper, last_action: Optional[int] = None):
            self.state = state
            self.snap = snap
            self.visit_count = 0
            self.value_sum: float = 0.0
            self.expandable = stepper.valid_moves(state, snap).astype(np.uint8)
            self.last_action = last_action

    def _make_child(self, parent_state: np.ndarray, parent_snap: GameSnapshot, action: int) -> "GumbelMCTS._Node":
        ns, nsnap = self.stepper.step(parent_state, parent_snap, action)
        return self._Node(ns, nsnap, self.stepper, last_action=action)

    def _q(self, node: "GumbelMCTS._Node") -> float:
        return 0.0 if node.visit_count == 0 else (node.value_sum / node.visit_count)

    def _simulate_from_node(self, node: "GumbelMCTS._Node") -> float:
        """
        Rollout losowy aż do terminala; zwraca wynik z perspektywy GRACZA NA RUCHU w tym węźle.
        Kluczowe: najpierw sprawdź terminal po 'last_action' (dziecko korzenia).
        """
        v, term = self.stepper.value_and_terminated(node.state, node.snap, last_action=node.last_action)
        if term:
            # 'v' oceniony z perspektywy ostatniego ruchu; gracz na ruchu w node to przeciwnik:
            return -float(v)

        st = node.state.copy()
        sn = GameSnapshot(**vars(node.snap))
        start_to_play = sn.to_play

        while True:
            valid = self.stepper.valid_moves(st, sn)
            moves = np.where(valid == 1)[0]
            a = int(self.rng.choice(moves))
            st, sn = self.stepper.step(st, sn, a)

            v, term = self.stepper.value_and_terminated(st, sn, last_action=a)
            if term:
                if v == 0:
                    return 0.0
                last_mover = -sn.to_play  # ten, kto właśnie zagrał
                return 1.0 if last_mover == start_to_play else -1.0

    def _backup_from_node(self, node: "GumbelMCTS._Node", value_for_node_player: float) -> None:
        """Lokalny backup jak w SH (bez łańcucha do korzenia klasycznego MCTS)."""
        node.visit_count += 1
        node.value_sum += value_for_node_player

    # ----- narzędzia korzenia -----

    def _make_root_logits(self, valid: np.ndarray, policy_logits: Optional[np.ndarray]) -> np.ndarray:
        if policy_logits is None:
            logits = np.full(self.game.action_size, -np.inf, dtype=np.float32)
            logits[valid == 1] = 0.0
        else:
            logits = np.array(policy_logits, dtype=np.float32).copy()
            logits[valid == 0] = -np.inf

        # temperatura na logitach
        T = max(1e-6, self.root_logits_temperature)
        if T != 1.0:
            # softmax^{-1}(softmax(logits)/T) ≈ logits / T dla wykorzystania w kluczach
            logits = logits / T

        # Dirichlet (opcjonalnie) — domieszka na priory w korzeniu
        if self.root_dirichlet_epsilon is not None and self.root_dirichlet_epsilon > 0.0:
            eps = float(self.root_dirichlet_epsilon)
            alpha = float(self.root_dirichlet_alpha)
            legal = (valid == 1)
            k = int(legal.sum())
            if k > 0:
                noise = np.zeros_like(logits, dtype=np.float32)
                noise_vals = self.rng.dirichlet([alpha] * k).astype(np.float32)
                noise[legal] = noise_vals
                # przejście do „logitów” – pracujemy addytywnie; użyjmy log na bezpiecznej masie
                # (to tylko heurystyczna domieszka; ważne by zachować maskę)
                base = np.zeros_like(logits, dtype=np.float32)
                base[legal] = 1.0 / k
                mix = (1.0 - eps) * base + eps * noise
                # w przestrzeni logitów: log(prob) (niezależnie od skali, zachowujemy kolejność)
                with np.errstate(divide="ignore"):
                    logits[legal] = np.log(mix[legal])
        return logits

    def _choose_m(self, num_legal: int) -> int:
        if self.m is not None:
            return max(1, min(self.m, num_legal))
        return max(1, min(self.n, num_legal))

    def _label_action(self, a: int) -> Tuple[str, str]:
        if a < self.game.board_actions:
            r = a // self.game.column_count
            c = a % self.game.column_count
            return "BOARD", f"{r+1} {c+1}"
        if hasattr(self.game, "SWAP_CHOOSE_WHITE") and a == self.game.SWAP_CHOOSE_WHITE:
            return "SWAP2", "CHOOSE_WHITE"
        if hasattr(self.game, "SWAP_CHOOSE_BLACK") and a == self.game.SWAP_CHOOSE_BLACK:
            return "SWAP2", "CHOOSE_BLACK"
        if hasattr(self.game, "SWAP_PLACE_TWO") and a == self.game.SWAP_PLACE_TWO:
            return "SWAP2", "PLACE_TWO"
        return "SPECIAL", str(a)

    # ----- szybkie taktyki (opcjonalne) -----

    def _find_my_winning_moves_now(self, state: np.ndarray, snap: GameSnapshot) -> List[int]:
        """Lista ruchów bieżącego gracza, które wygrywają natychmiast."""
        res: List[int] = []
        valid = self.stepper.valid_moves(state, snap)
        for a in np.where(valid == 1)[0]:
            st2, sn2 = self.stepper.step(state, snap, int(a))
            v, term = self.stepper.value_and_terminated(st2, sn2, last_action=int(a))
            if term and v > 0:
                res.append(int(a))
        return res

    def _find_opponent_winning_moves_next(self, state: np.ndarray, snap: GameSnapshot) -> List[int]:
        """
        Sprawdza, czy PRZECIWNIK (−snap.to_play) ma mata w 1 z bieżącej pozycji (gdy dostanie ruch).
        Zwraca listę pól, na które przeciwnik wygra od razu.
        """
        threats: List[int] = []
        tmp = GameSnapshot(**vars(snap))
        tmp.to_play = -snap.to_play  # udajemy, że przeciwnik jest na ruchu
        valid_opp = self.stepper.valid_moves(state, tmp)
        for b in np.where(valid_opp == 1)[0]:
            st2, sn2 = self.stepper.step(state, tmp, int(b))
            v, term = self.stepper.value_and_terminated(st2, sn2, last_action=int(b))
            if term and v > 0:
                threats.append(int(b))
        return threats
