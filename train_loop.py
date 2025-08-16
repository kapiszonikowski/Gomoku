import os
import time
import math
import random
import shutil
import multiprocessing as mp
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from Gomoku import Gomoku
from Gomoku_NN import GomokuNet
from GUMBEL import GumbelMCTS

# ===================== Torch setup =====================

def configure_torch_for_speed():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Ampere/Ada: TF32 szybciej
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.set_num_threads(max(1, mp.cpu_count() - 1))

def safe_compile(model: torch.nn.Module) -> torch.nn.Module:
    """
    Na Windows, gdy brak kompilatora (cl.exe/clang-cl), torch.compile wywali:
    'Compiler: cl is not found'. W takim wypadku pomijamy kompilację i zostajemy przy eager.
    """
    if not hasattr(torch, "compile"):
        return model
    # Windows: sprawdź czy mamy kompilator (MSVC 'cl' albo ustawiony CXX)
    if os.name == "nt":
        has_cl = shutil.which("cl") is not None
        has_cxx = bool(os.environ.get("CXX", "").strip())
        if not (has_cl or has_cxx):
            print("[INFO] Pomijam torch.compile: brak kompilatora (cl/clang-cl) w PATH.")
            return model
    try:
        # bezpieczny profil; jeśli coś pójdzie nie tak -> fallback w except
        return torch.compile(model, mode="max-autotune-no-cudagraphs")
    except Exception as e:
        print(f"[WARN] torch.compile wyłączony (fallback eager): {e}")
        return model

# ===================== UI: wykres postępu =====================

def init_progress_window():
    plt.ion()
    fig, (ax_loss, ax_info) = plt.subplots(2, 1, figsize=(6, 8))
    loss_line, = ax_loss.plot([], [], label="loss")
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)
    ax_info.axis("off")
    info_text = ax_info.text(0.05, 0.95, "", va="top")
    plt.tight_layout()
    plt.show(block=False)
    return fig, loss_line, ax_loss, info_text

def update_progress(fig, loss_line, ax_loss, info_text,
                    iterations, losses, save_count, start_time,
                    replay_size, last_eval):
    loss_line.set_data(iterations, losses)
    ax_loss.relim()
    ax_loss.autoscale_view()
    elapsed = time.time() - start_time
    info = (
        f"Iteration: {iterations[-1]}\n"
        f"Saved models: {save_count}\n"
        f"Elapsed: {elapsed/60:.1f} min\n"
        f"Replay size: {replay_size}"
    )
    if last_eval is not None:
        info += f"\nLast eval win rate: {last_eval:.2f}"
    info_text.set_text(info)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

# ===================== Augmentacje / utils =====================

def augment_symmetries(board: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = board.shape[0]
    board_moves = policy[:-3].reshape(N, N)
    special = policy[-3:]
    boards, policies = [], []
    for k in range(4):
        rb = np.rot90(board, k)
        rp = np.rot90(board_moves, k)
        boards.append(rb)
        policies.append(np.concatenate([rp.reshape(-1), special]))
        fb = np.fliplr(rb)
        fp = np.fliplr(rp)
        boards.append(fb)
        policies.append(np.concatenate([fp.reshape(-1), special]))
    return np.stack(boards), np.stack(policies)

def temperature_by_move(move: int, base: float = 0.1, early: float = 1.0, moves: int = 8) -> float:
    return early if move < moves else base

def logits_top_k(logits: np.ndarray, k: Optional[int] = None, min_p: float = 1e-3) -> np.ndarray:
    probs = torch.softmax(torch.as_tensor(logits), dim=0).cpu().numpy()
    idx = np.argsort(probs)[::-1]
    if k is not None:
        idx = idx[:k]
    idx = [i for i in idx if probs[i] >= min_p]
    masked = np.full_like(logits, -np.inf, dtype=np.float32)
    masked[idx] = logits[idx]
    return masked

# ===================== Replay =====================

@dataclass
class ReplayBuffer:
    capacity: int
    buffer: deque = None

    def __post_init__(self):
        self.buffer = deque(maxlen=self.capacity)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        s, p, v = zip(*batch)
        return np.array(s), np.array(p), np.array(v, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

# ===================== Self-play =====================

def self_play_worker(worker_id: int, net: GomokuNet, replay_queue: mp.Queue,
                     sim_schedule: list[tuple[int, int]], top_k: int,
                     min_p: float, stop_flag: mp.Event):
    configure_torch_for_speed()
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()
    game = Gomoku()
    rng = np.random.default_rng(seed=worker_id)
    games_played = 0

    while not stop_flag.is_set():
        state = game.get_initial_state()
        moves = []
        stats = []
        done = False
        while not done:
            sims = 16
            for threshold, n in sim_schedule:
                if games_played >= threshold:
                    sims = n
            gumbel = GumbelMCTS(game, num_simulations=sims, NN=True, net=net, device=device)
            with torch.no_grad():
                out = net(state)
                logits = out["policy_logits"].squeeze(0).cpu().numpy()
            filtered = logits_top_k(logits, k=top_k, min_p=min_p)
            probs = gumbel.search(state, policy_logits=filtered)
            temp = temperature_by_move(game.move_count)
            if temp != 1.0:
                probs = probs ** (1.0 / temp)
                probs = probs / probs.sum()
            action = rng.choice(len(probs), p=probs)
            moves.append((state.copy(), probs))
            state = game.get_next_state(state, action, 0)
            value, done = game.get_value_and_terminated(state, action)
            stats.append((action, value))
        # backpropagate game result
        result = value
        for (s, pi), _ in zip(moves, stats):
            z = result if game.to_play == 1 else -result
            boards, policies = augment_symmetries(s, pi)
            for b, p in zip(boards, policies):
                replay_queue.put((b, p, z))
        games_played += 1

# ===================== Ewaluacja =====================

def evaluate_model(net: GomokuNet, games: int = 10, sims: int = 256,
                   record_dir: Optional[str] = None) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()
    scores = []
    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
    for g in range(games):
        game = Gomoku()
        state = game.get_initial_state()
        done = False
        moves: list[int] = []
        while not done:
            mcts = GumbelMCTS(game, num_simulations=sims, NN=True, net=net, device=device)
            with torch.no_grad():
                logits = net(state)["policy_logits"].squeeze(0).cpu().numpy()
            probs = mcts.search(state, policy_logits=logits)
            action = int(np.argmax(probs))
            moves.append(action)
            state = game.get_next_state(state, action, 0)
            value, done = game.get_value_and_terminated(state, action)
        scores.append(value)
        if record_dir is not None:
            path = os.path.join(record_dir, f"game_{int(time.time())}_{g}.txt")
            with open(path, "w") as f:
                f.write(",".join(map(str, moves)))
    return float(sum(scores)) / len(scores)

# ===================== Trening =====================

def train_loop(
    total_iterations: int = 100000,
    replay_size: int = 200000,
    batch_size: int = 512,
    save_every: float = 3600.0,
    num_workers: Optional[int] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_for_speed()

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    net = GomokuNet().to(device)
    net = safe_compile(net)  # <— bezpieczny fallback na Windows bez cl.exe
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    replay = ReplayBuffer(replay_size)
    replay_queue: mp.Queue = mp.Queue(maxsize=10000)
    stop_flag = mp.Event()

    sim_schedule = [(0, 16), (10000, 32), (20000, 64)]
    workers = [mp.Process(target=self_play_worker, args=(i, net, replay_queue,
                                                         sim_schedule, 32, 1e-3, stop_flag))
               for i in range(num_workers)]
    for w in workers:
        w.start()

    next_save = time.time() + save_every

    # Nowe API AMP (deprecacja starego cuda.amp.GradScaler)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    fig, loss_line, ax_loss, info_text = init_progress_window()
    losses: list[float] = []
    iterations_list: list[int] = []
    save_count = 0
    start_time = time.time()
    last_eval: Optional[float] = None

    def policy_loss(logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=0.05)

    # Warmup: doładuj trochę danych, aby uniknąć „sample larger than population”
    warmup_min = max(batch_size * 4, 4096)
    pbar = tqdm(total=total_iterations)
    iteration = 0
    try:
        while iteration < total_iterations:
            # pobierz z kolejek do replaya
            while len(replay) < warmup_min:
                try:
                    s, p, v = replay_queue.get(timeout=1.0)
                    replay.add(s, p, v)
                except Exception:
                    if stop_flag.is_set():
                        break

            if len(replay) < batch_size:
                continue  # jeszcze chwilę!

            # standardowy batch
            states, policies, values = replay.sample(batch_size)
            # targets: indeks ruchu (argmax po polityce)
            policy_idx = np.argmax(policies, axis=1).astype(np.int64)

            # tensory
            states_t = states  # numpy -> konwersja zrobi model
            policy_t = torch.from_numpy(policy_idx)
            value_t = torch.from_numpy(values).float()

            if torch.cuda.is_available():
                policy_t = policy_t.pin_memory().to(device, non_blocking=True)
                value_t = value_t.pin_memory().to(device, non_blocking=True)
            else:
                policy_t = policy_t.to(device)
                value_t = value_t.to(device)

            # forward + loss (AMP)
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                out = net(states_t)  # model sam przyjmie numpy i skonwertuje
                logits = out["policy_logits"]
                val = out["value"].squeeze(1) if out["value"].dim() == 2 else out["value"]
                p_loss = policy_loss(logits, policy_t)
                v_loss = F.mse_loss(val, value_t)
                loss = p_loss + v_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iteration += 1

            # checkpoint
            if time.time() >= next_save:
                torch.save(net.state_dict(), f"model_{int(time.time())}.pt")
                save_count += 1
                next_save = time.time() + save_every

            # szybka ewaluacja co 1000 iteracji
            if iteration % 1000 == 0:
                score = evaluate_model(net, games=2, sims=300, record_dir="test_games")
                last_eval = score
                pbar.write(f"Evaluation win rate: {score:.2f}")

            # log
            losses.append(float(loss.item()))
            iterations_list.append(iteration)
            update_progress(fig, loss_line, ax_loss, info_text,
                            iterations_list, losses, save_count, start_time,
                            len(replay), last_eval)
            pbar.update(1)
            pbar.set_postfix({"loss": float(loss.item())})

    finally:
        pbar.close()
        plt.ioff()
        plt.close(fig)
        stop_flag.set()
        for w in workers:
            w.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_loop()
