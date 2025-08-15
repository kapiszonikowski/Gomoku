import sys
import random
import pygame

# ---------- Konfiguracja ----------
BOARD_SIZE = 15
CELL = 40
MARGIN = 50
PANEL_W = 320
BOARD_PIX = CELL * BOARD_SIZE
W = MARGIN * 2 + BOARD_PIX + PANEL_W
H = MARGIN * 2 + BOARD_PIX

BG_COL = (240, 217, 181)
GRID_COL = (70, 70, 70)
BLACK_COL = (20, 20, 20)
WHITE_COL = (235, 235, 235)
HILITE_COL = (230, 90, 50)
TEXT_COL = (30, 30, 30)
BTN_BG = (245, 245, 245)
BTN_BORDER = (160, 160, 160)
BTN_BG_HOT = (230, 230, 230)

EMPTY, BLACK, WHITE = 0, 1, 2

pygame.init()
pygame.display.set_caption(f"Gomoku {BOARD_SIZE}×{BOARD_SIZE} — Python/pygame")

# --- Okno i canvas + pełny ekran ---
window = pygame.display.set_mode((W, H))
screen = pygame.Surface((W, H)).convert()
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
font_small = pygame.font.SysFont("Arial", 15)
font_big = pygame.font.SysFont("Arial", 22, bold=True)

fullscreen = False
offset = [0, 0]

def set_fullscreen(on: bool):
    global window, fullscreen, offset
    fullscreen = on
    if on:
        info = pygame.display.Info()
        window = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
        offset[0] = (info.current_w - W) // 2
        offset[1] = (info.current_h - H) // 2
    else:
        window = pygame.display.set_mode((W, H))
        offset[0], offset[1] = 0, 0

# ---------- Rysowanie planszy ----------
def cell_center(r, c):
    x = MARGIN + c * CELL + CELL // 2
    y = MARGIN + r * CELL + CELL // 2
    return x, y

def draw_board():
    screen.fill(BG_COL)
    left, top = MARGIN, MARGIN
    right, bottom = MARGIN + BOARD_PIX, MARGIN + BOARD_PIX
    pygame.draw.rect(screen, GRID_COL, (left, top, BOARD_PIX, BOARD_PIX), 2)
    for i in range(1, BOARD_SIZE):
        y = top + i * CELL
        x = left + i * CELL
        pygame.draw.line(screen, GRID_COL, (left, y), (right, y), 1)
        pygame.draw.line(screen, GRID_COL, (x, top), (x, bottom), 1)

def draw_stone(r, c, color, last=False):
    x, y = cell_center(r, c)
    col = BLACK_COL if color == BLACK else WHITE_COL
    pygame.draw.circle(screen, col, (x, y), CELL//2 - 6)
    pygame.draw.circle(screen, (0,0,0) if color == WHITE else (255,255,255), (x, y), CELL//2 - 6, 1)
    if last:
        pygame.draw.circle(screen, HILITE_COL, (x, y), 6)

def draw_winline(positions):
    if not positions:
        return
    positions = sorted(positions)
    x1, y1 = cell_center(*positions[0])
    x2, y2 = cell_center(*positions[-1])
    pygame.draw.line(screen, HILITE_COL, (x1, y1), (x2, y2), 6)

# ---------- Logika gry ----------
class Swap2State:
    """
    Fazy globalne:
      - MODE_SELECT: wybór 'CLASSIC' / 'SWAP2'
      - (SWAP2) P1_OPEN -> P2_CHOOSE -> (P2_PLACE_W | P2_PLACE_BW -> P1_PICK_COLOR) -> PLAY -> GAME_OVER
      - (CLASSIC) PLAY -> GAME_OVER
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.last_move = None
        self.win_line = []
        self.exact_five = True

        # tryb i fazy
        self.mode = None        # 'CLASSIC' albo 'SWAP2'
        self.phase = "MODE_SELECT"

        # SWAP2:
        self.open_seq = [BLACK, WHITE, BLACK]
        self.open_idx = 0
        self.p2_bw_to_place = []

        # kolory graczy
        self.p1_color = None
        self.p2_color = None
        self.current_color = None

    # --- pomocnicze ---
    def inside(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def place(self, r, c, color):
        if not self.inside(r,c) or self.board[r][c] != EMPTY:
            return False
        self.board[r][c] = color
        self.last_move = (r, c, color)
        return True

    def count_dir(self, r, c, color, dr, dc):
        k, pos = 0, []
        rr, cc = r + dr, c + dc
        while self.inside(rr, cc) and self.board[rr][cc] == color:
            pos.append((rr, cc))
            k += 1
            rr += dr
            cc += dc
        return k, pos

    def check_win(self, r, c, color):
        for dr, dc in [(1,0), (0,1), (1,1), (1,-1)]:
            k1, pos1 = self.count_dir(r, c, color, dr, dc)
            k2, pos2 = self.count_dir(r, c, color, -dr, -dc)
            total = 1 + k1 + k2
            line = list(reversed(pos2)) + [(r, c)] + pos1
            if self.exact_five:
                if total == 5: return line
            else:
                if total >= 5: return line
        return []

    def board_full(self):
        return all(all(x != EMPTY for x in row) for row in self.board)

    # --- wybór trybu ---
    def choose_mode_classic(self):
        self.mode = "CLASSIC"
        # losowanie koloru graczy; czarne zaczynają
        if random.choice([True, False]):
            self.p1_color, self.p2_color = BLACK, WHITE
        else:
            self.p1_color, self.p2_color = WHITE, BLACK
        self.current_color = BLACK
        self.phase = "PLAY"

    def choose_mode_swap2(self):
        self.mode = "SWAP2"
        self.open_idx = 0
        self.p2_bw_to_place = []
        self.p1_color = None
        self.p2_color = None
        self.current_color = None
        self.phase = "P1_OPEN"

    # --- zdarzenia gry ---
    def click_board(self, r, c):
        if self.phase in ("MODE_SELECT", "GAME_OVER"):
            return

        if self.mode == "SWAP2":
            if self.phase == "P1_OPEN":
                color = self.open_seq[self.open_idx]
                if self.place(r, c, color):
                    self.open_idx += 1
                    if self.open_idx >= len(self.open_seq):
                        self.phase = "P2_CHOOSE"
                return

            if self.phase == "P2_PLACE_W":
                if self.place(r, c, WHITE):
                    self.p1_color = BLACK
                    self.p2_color = WHITE
                    self.current_color = BLACK
                    self.phase = "PLAY"
                return

            if self.phase == "P2_PLACE_BW":
                if not self.p2_bw_to_place:
                    return
                color_to_place = self.p2_bw_to_place[0]
                if self.place(r, c, color_to_place):
                    self.p2_bw_to_place.pop(0)
                    if not self.p2_bw_to_place:
                        self.phase = "P1_PICK_COLOR"
                return

        # PLAY (oba tryby)
        if self.phase == "PLAY":
            color = self.current_color
            if self.place(r, c, color):
                wl = self.check_win(r, c, color)
                if wl:
                    self.win_line = wl
                    self.phase = "GAME_OVER"
                    return
                if self.board_full():
                    self.phase = "GAME_OVER"
                    return
                self.current_color = WHITE if self.current_color == BLACK else BLACK

    # --- akcje Swap2 ---
    def choose_option_take_black(self):
        if not (self.mode == "SWAP2" and self.phase == "P2_CHOOSE"):
            return
        self.p2_color = BLACK
        self.p1_color = WHITE
        self.current_color = WHITE  # po swapie ruszają białe
        self.phase = "PLAY"

    def choose_option_take_white_plusW(self):
        if not (self.mode == "SWAP2" and self.phase == "P2_CHOOSE"):
            return
        self.phase = "P2_PLACE_W"

    def choose_option_place_BW(self):
        if not (self.mode == "SWAP2" and self.phase == "P2_CHOOSE"):
            return
        self.p2_bw_to_place = [BLACK, WHITE]
        self.phase = "P2_PLACE_BW"

    def toggle_order_bw(self):
        if self.mode == "SWAP2" and self.phase == "P2_PLACE_BW" and len(self.p2_bw_to_place) >= 2:
            self.p2_bw_to_place[0], self.p2_bw_to_place[1] = self.p2_bw_to_place[1], self.p2_bw_to_place[0]

    def p1_pick_black(self):
        if not (self.mode == "SWAP2" and self.phase == "P1_PICK_COLOR"): return
        self.p1_color, self.p2_color = BLACK, WHITE
        self.current_color = BLACK
        self.phase = "PLAY"

    def p1_pick_white(self):
        if not (self.mode == "SWAP2" and self.phase == "P1_PICK_COLOR"): return
        self.p1_color, self.p2_color = WHITE, BLACK
        self.current_color = WHITE
        self.phase = "PLAY"

# ---------- Panel i przyciski ----------
class Button:
    def __init__(self, rect, label, action):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.action = action
    def draw(self, surf, hot=False):
        bg = BTN_BG_HOT if hot else BTN_BG
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, BTN_BORDER, self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, TEXT_COL)
        surf.blit(txt, (self.rect.x + 10, self.rect.y + (self.rect.h - txt.get_height())//2))
    def hit(self, pos): return self.rect.collidepoint(pos)

def build_buttons(state):
    x0 = W - PANEL_W + 20
    y = H - 186*2
    btns = []
    def add(label, cb):
        nonlocal y
        r = (x0, y, PANEL_W - 40, 38)
        btns.append(Button(r, label, cb))
        y += 46

    # górna sekcja zależna od fazy
    if state.phase == "MODE_SELECT":
        add("Tryb klasyczny (losowanie koloru) [1]", state.choose_mode_classic)
        add("Tryb Swap2 [2]", state.choose_mode_swap2)
    elif state.mode == "SWAP2" and state.phase == "P2_CHOOSE":
        add("Opcja 1: Weź czarne (Swap)", state.choose_option_take_black)
        add("Opcja 2: Weź białe + postaw 1W", state.choose_option_take_white_plusW)
        add("Opcja 3: Postaw B i W", state.choose_option_place_BW)
    elif state.mode == "SWAP2" and state.phase == "P1_PICK_COLOR":
        add("P1 wybiera: Czarne", state.p1_pick_black)
        add("P1 wybiera: Białe", state.p1_pick_white)

    # dolny blok – kotwiczony do dołu, bez nachodzenia
    bottom_buttons = []
    def add_bottom(label, cb): bottom_buttons.append((label, cb))

    add_bottom(("Tryb: \"dokładnie 5\"" if state.exact_five else "Tryb: \"≥ 5\""),
               lambda: setattr(state, "exact_five", not state.exact_five))
    add_bottom("Pełny ekran [F11]", lambda: set_fullscreen(not fullscreen))
    add_bottom("Nowa gra [N]", state.reset)
    add_bottom("Wyjście [Esc]", lambda: sys.exit(0))

    # wyznacz start Y tak, by nie kolidować z górną sekcją
    used_top = y
    needed_bottom_h = len(bottom_buttons) * 46 + 16
    y = max(used_top + 12, H - needed_bottom_h - 20)

    for label, cb in bottom_buttons:
        add(label, cb)

    return btns

def draw_panel(state, buttons, mouse_pos):
    panel_rect = pygame.Rect(W - PANEL_W, 0, PANEL_W, H)
    pygame.draw.rect(screen, (252,252,252), panel_rect)
    pygame.draw.line(screen, (200,200,200), (W - PANEL_W, 0), (W - PANEL_W, H), 1)

    title = font_big.render(f"Gomoku {BOARD_SIZE}×{BOARD_SIZE}", True, TEXT_COL)
    screen.blit(title, (W - PANEL_W + 16, 12))

    y = 50
    def put(line, bold=False):
        nonlocal y
        f = font_big if bold else font
        txt = f.render(line, True, TEXT_COL)
        screen.blit(txt, (W - PANEL_W + 16, y))
        y += txt.get_height() + 4

    put("Status:", bold=True)
    if state.phase == "MODE_SELECT":
        put("Wybierz tryb gry (1/2).")
        put("Klasyczny: losowanie koloru; czarne rozpoczynają.")
        put("Swap2: otwarcie równoważące.")
    elif state.mode == "SWAP2":
        if state.phase == "P1_OPEN":
            seq_names = ["Czarne", "Białe", "Czarne"]
            put(f"Otwarcie P1: postaw {seq_names[state.open_idx]}")
            put("Klikaj w kratki.")
        elif state.phase == "P2_CHOOSE":
            put("P2: wybierz opcję Swap2 (poniżej).")
        elif state.phase == "P2_PLACE_W":
            put("P2 (Białe): postaw 1 dodatkowy biały.")
        elif state.phase == "P2_PLACE_BW":
            nxt = "Czarne" if state.p2_bw_to_place and state.p2_bw_to_place[0]==BLACK else "Białe"
            put(f"P2: postaw {nxt}.")
            put("PPM / Spacja: zmiana kolejnego koloru.")
        elif state.phase == "P1_PICK_COLOR":
            put("P1: wybierz kolor (przyciski poniżej).")
        elif state.phase == "PLAY":
            col = "Czarne" if state.current_color == BLACK else "Białe"
            put(f"Ruch: {col}.")
            put(f"Warunek: {'dokładnie 5' if state.exact_five else '≥ 5'}.")
        elif state.phase == "GAME_OVER":
            if state.win_line:
                winner = "Czarne" if state.board[state.win_line[0][0]][state.win_line[0][1]] == BLACK else "Białe"
                put(f"Koniec: wygrywają {winner}!", bold=True)
            else:
                put("Koniec: remis (plansza pełna).", bold=True)
    else:  # CLASSIC
        if state.phase == "PLAY":
            col = "Czarne" if state.current_color == BLACK else "Białe"
            who_p1 = "Czarne" if state.p1_color == BLACK else "Białe"
            put(f"Tryb klasyczny. P1: {who_p1}.")
            put(f"Ruch: {col}. Warunek: {'dokładnie 5' if state.exact_five else '≥ 5'}.")
        elif state.phase == "GAME_OVER":
            if state.win_line:
                winner = "Czarne" if state.board[state.win_line[0][0]][state.win_line[0][1]] == BLACK else "Białe"
                put(f"Koniec: wygrywają {winner}!", bold=True)
            else:
                put("Koniec: remis (plansza pełna).", bold=True)

    put("")
    put("Skróty:", bold=True)
    put("[1]/[2] — wybór trybu na starcie")
    put("[E] — warunek zwycięstwa")
    put("[F11] — pełny ekran")
    put("[N] — nowa gra")
    put("[Esc] — wyjście")

    for b in buttons:
        b.draw(screen, hot=b.hit(mouse_pos))

# ---------- Wejście myszy ----------
def nearest_cell(mx, my):
    left, top = MARGIN, MARGIN
    if not (left <= mx < left + BOARD_PIX and top <= my < top + BOARD_PIX):
        return None
    c = int((mx - left) // CELL)
    r = int((my - top) // CELL)
    return (r, c)

# ---------- Pętla główna ----------
def main():
    state = Swap2State()
    running = True
    while running:
        clock.tick(60)
        mouse_win = pygame.mouse.get_pos()
        mouse_pos = (mouse_win[0] - offset[0], mouse_win[1] - offset[1])

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_e:
                    state.exact_five = not state.exact_five
                elif e.key == pygame.K_n:
                    state.reset()
                elif e.key == pygame.K_SPACE:
                    state.toggle_order_bw()
                elif e.key == pygame.K_F11:
                    set_fullscreen(fullscreen) if False else set_fullscreen(not fullscreen)
                elif e.key == pygame.K_1 and state.phase == "MODE_SELECT":
                    state.choose_mode_classic()
                elif e.key == pygame.K_2 and state.phase == "MODE_SELECT":
                    state.choose_mode_swap2()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 3:
                    state.toggle_order_bw()
                if e.button == 1:
                    buttons = build_buttons(state)
                    if any(b.hit(mouse_pos) and (b.action() or True) for b in buttons):
                        pass
                    else:
                        rc = nearest_cell(*mouse_pos)
                        if rc is not None:
                            state.click_board(*rc)

        # Rysowanie
        draw_board()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                v = state.board[r][c]
                if v == EMPTY: continue
                last = state.last_move == (r, c, v)
                draw_stone(r, c, v, last=last)
        draw_winline(state.win_line)

        buttons = build_buttons(state)
        draw_panel(state, buttons, mouse_pos)

        window.fill((0,0,0))
        window.blit(screen, offset)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
