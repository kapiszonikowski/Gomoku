#!/usr/bin/env python
# coding: utf-8

import numpy as np


class Gomoku:
    def __init__(self, size: int = 15, win_length: int = 5, exact_five: bool = False, use_swap2: bool = True):
        """
        Minimalny silnik Gomoku (NumPy only) z wariantem Swap2 i czytelną logiką zgodną z "gomoku_rules".

        Args:
            size: rozmiar planszy (size x size)
            win_length: długość linii potrzebna do wygranej (domyślnie 5)
            exact_five: jeśli True – zwycięża dokładnie 5; dłuższe linie nie liczą się (wariant turniejowy). Jeśli False – >=5 wygrywa (freestyle).
            use_swap2: włącza protokół otwarcia Swap2 z trzema dodatkowymi indeksami akcji (size*size..size*size+2).
        """
        self.row_count = size
        self.column_count = size
        self.board_actions = self.row_count * self.column_count
        self.action_size = self.board_actions + 3  # 3 akcje specjalne dla Swap2
        # Stałe akcji specjalnych (dla 15x15: 225..227)
        self.SWAP_CHOOSE_WHITE = self.board_actions + 0  # wybierz BIAŁE (patrz: znaczenie zależne od fazy)
        self.SWAP_CHOOSE_BLACK = self.board_actions + 1  # wybierz CZARNE
        self.SWAP_PLACE_TWO   = self.board_actions + 2  # wybierz wariant „połóż dwa kamienie (W, potem B)”

        self.win_length = win_length
        self.exact_five = exact_five
        self.use_swap2 = use_swap2

        # Stan runtime
        self.to_play = 1            # na ruchu: 1 (Czarne) lub -1 (Białe)
        self.move_count = 0
        # Fazy Swap2:
        # 0 = normalna gra
        # 1 = Proposer A układa 3 kamienie (B, W, B) – pojedynczo w zewnętrznej pętli
        # 2 = Chooser B wybiera: CHOOSE_BLACK / CHOOSE_WHITE(+1 biały) / PLACE_TWO (W potem B)
        # 3 = Chooser B kładzie dwa kamienie (W, potem B) – wymuszone kolory
        # 4 = Proposer A wybiera kolor (CHOOSE_WHITE / CHOOSE_BLACK); po wyborze rusza PRZECIWNY kolor
        # 5 = Chooser B (wybrał bycie białymi) dokłada dodatkowy biały kamień – po nim na ruchu czarne
        self.swap_phase = 0
        self.swap_substep = 0  # licznik kamieni wewnątrz faz 1/3/5

        # Przypisanie kolorów do graczy A/B (po rozstrzygnięciu Swap2)
        self.playerA_color = None  # 1 lub -1
        self.playerB_color = None

    def get_initial_state(self):
        state = np.zeros((self.row_count, self.column_count), dtype=np.int8)
        self.to_play = 1
        self.move_count = 0
        self.playerA_color = None
        self.playerB_color = None
        if self.use_swap2:
            self.swap_phase = 1
            self.swap_substep = 0
        else:
            self.swap_phase = 0
            self.swap_substep = 0
        return state

    # === API gry ===
    def get_next_state(self, state: np.ndarray, action: int, _player_arg_ignored: int):
        """
        Zastosuj akcję. Dla pól planszy stawiamy kamień koloru self.to_play (w niektórych fazach narzucony).
        Parametr player jest ignorowany – logika steruje kolorem wg faz Swap2.
        Zwraca zmodyfikowany stan.
        """
        if action < 0 or action >= self.action_size:
            raise ValueError("Action out of range")

        # --- Ruch na planszy ---
        if action < self.board_actions:
            r = action // self.column_count
            c = action % self.column_count
            state[r, c] = self.to_play

            if self.swap_phase == 1:
                # A układa 3 kamienie: kolejność B (1) -> W (-1) -> B (1)
                self.swap_substep += 1
                if self.swap_substep < 3:
                    self.to_play = -self.to_play  # przełącz kolor dla kolejnego z 3 kamieni
                else:
                    # po trzech kamieniach – przejście do wyboru opcji przez B
                    self.swap_phase = 2
                    self.swap_substep = 0
                    # to_play pozostaje bez znaczenia, czekamy na akcję specjalną

            elif self.swap_phase == 3:
                # B kładzie dwa kamienie: najpierw BIAŁY (-1), potem CZARNY (1)
                self.swap_substep += 1
                if self.swap_substep == 1:
                    self.to_play = 1  # drugi kamień będzie czarny
                elif self.swap_substep == 2:
                    # przejście do wyboru koloru przez A
                    self.swap_phase = 4
                    self.swap_substep = 0
                else:
                    raise RuntimeError("Invalid swap_substep in phase 3")

            elif self.swap_phase == 5:
                # B wybrał grę białymi i dokłada jeden dodatkowy biały – po nim gra wraca do normalnej fazy
                # (ten kamień już położony, więc przechodzimy do normalnej gry – na ruchu CZARNE)
                self.swap_phase = 0
                self.to_play = 1

            else:
                # normalne naprzemienne ruchy
                self.to_play = -self.to_play

            self.move_count += 1
            return state

        # --- Akcje specjalne Swap2 ---
        if not self.use_swap2 or self.swap_phase == 0:
            raise ValueError("Special swap action when swap2 is not active")

        if self.swap_phase == 2:
            if action == self.SWAP_CHOOSE_BLACK:
                # B wybiera CZARNE -> przypisz kolory i zaczyna BIAŁY
                self.playerB_color = 1
                self.playerA_color = -1
                self.swap_phase = 0
                self.to_play = -1
            elif action == self.SWAP_CHOOSE_WHITE:
                # B wybiera BIAŁE i musi dołożyć dodatkowy BIAŁY kamień
                self.playerB_color = -1
                self.playerA_color = 1
                self.swap_phase = 5
                self.swap_substep = 0
                self.to_play = -1  # biały dołoży 1 kamień
            elif action == self.SWAP_PLACE_TWO:
                # B układa 2 kamienie: W, potem B
                self.swap_phase = 3
                self.swap_substep = 0
                self.to_play = -1
            else:
                raise ValueError("Invalid special action in phase 2")
            return state

        if self.swap_phase == 4:
            if action == self.SWAP_CHOOSE_BLACK:
                # A wybiera CZARNE -> po wyborze na ruchu BIAŁY
                self.playerA_color = 1
                self.playerB_color = -1
                self.swap_phase = 0
                self.to_play = -1
            elif action == self.SWAP_CHOOSE_WHITE:
                # A wybiera BIAŁE -> po wyborze na ruchu CZARNY
                self.playerA_color = -1
                self.playerB_color = 1
                self.swap_phase = 0
                self.to_play = 1
            else:
                raise ValueError("Invalid special action in phase 4")
            return state

        # nie powinno się zdarzyć
        raise ValueError("Unknown special action / phase mismatch")

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """Zwraca maskę uint8 (długość = action_size) dozwolonych akcji.
        W fazach decyzji (2 i 4) legalne są TYLKO akcje specjalne, w fazach kładzenia (1,3,5) i normalnej (0) – tylko puste pola.
        """
        mask = np.zeros(self.action_size, dtype=np.uint8)

        if self.swap_phase in (2, 4):
            mask[self.SWAP_CHOOSE_WHITE] = 1
            mask[self.SWAP_CHOOSE_BLACK] = 1
            if self.swap_phase == 2:
                mask[self.SWAP_PLACE_TWO] = 1
            return mask

        # Fazy 0 (normalna), 1 (A kładzie 3), 3 (B kładzie 2), 5 (B dokłada 1 biały) -> tylko puste pola
        empty = (state.reshape(-1) == 0).astype(np.uint8)
        mask[:self.board_actions] = empty
        return mask

    # === Sprawdzanie zwycięstwa ===
    def _count_one_dir(self, state: np.ndarray, r: int, c: int, dr: int, dc: int, player: int) -> int:
        n = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < self.row_count and 0 <= cc < self.column_count and state[rr, cc] == player:
            n += 1
            rr += dr
            cc += dc
        return n

    def _end_cell(self, state: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
        player = state[r, c]
        rr, cc = r, c
        while 0 <= rr + dr < self.row_count and 0 <= cc + dc < self.column_count and state[rr + dr, cc + dc] == player:
            rr += dr
            cc += dc
        rr += dr
        cc += dc
        if 0 <= rr < self.row_count and 0 <= cc < self.column_count:
            return state[rr, cc]
        return 0

    def check_win(self, state: np.ndarray, action: int) -> bool:
        if action >= self.board_actions:
            return False  # akcje specjalne nie kończą gry
        r = action // self.column_count
        c = action % self.column_count
        player = state[r, c]
        if player == 0:
            return False

        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            forward = self._count_one_dir(state, r, c, dr, dc, player)
            backward = self._count_one_dir(state, r, c, -dr, -dc, player)
            total = 1 + forward + backward

            if not self.exact_five:
                if total >= self.win_length:
                    return True
            else:
                if total == self.win_length:
                    end1 = self._end_cell(state, r, c, dr, dc)
                    end2 = self._end_cell(state, r, c, -dr, -dc)
                    if end1 != player and end2 != player:
                        return True
        return False

    def get_value_and_terminated(self, state: np.ndarray, action: int):
        """Zwraca (value, terminated) z perspektywy gracza, który wykonał OSTATNI ruch.
        value = 1 jeśli wygrał, 0 w przeciwnym razie (remis lub gra trwa).
        Uwaga: w trakcie faz Swap2 (>0) ewentualne linie 5 nie kończą gry – najpierw domyka się protokół Swap2.
        """
        if self.swap_phase != 0:
            return 0, False

        if self.check_win(state, action):
            return 1, True
        if np.sum(state == 0) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: int) -> int:
        return -player

    # Pomocnicze: etykieta gracza (A/B), który jest na ruchu (do UI)
    def current_player_label(self) -> str:
        if self.swap_phase == 1:
            return "A"
        if self.swap_phase in (2, 3, 5):
            return "B"
        if self.swap_phase == 4:
            return "A"
        # normalna faza – na podstawie przypisanych kolorów
        if self.playerA_color is None or self.playerB_color is None:
            # na wszelki wypadek – przed rozstrzygnięciem Swap2 nie powinniśmy tu trafiać
            return "A" if self.to_play == 1 else "B"
        return "A" if self.to_play == self.playerA_color else "B"


