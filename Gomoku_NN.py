import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Blok rezydualny (bez zmian) ---
class ResidualBlock(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(C)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        return F.relu(out, inplace=True)

class GomokuNet(nn.Module):
    def __init__(self, board_size: int = 15, in_planes: int = 3,
                 channels: int = 64, blocks: int = 10):
        super().__init__()
        self.N = board_size
        self.board_actions = board_size * board_size
        self.total_actions = self.board_actions + 3  # +3 Swap2

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # trunk
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # --- Policy head ---
        self.pol_conv1 = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.pol_bn1   = nn.BatchNorm2d(2)
        self.pol_conv2 = nn.Conv2d(2, 1, kernel_size=1, bias=True)

        self.pol_gap   = nn.AdaptiveAvgPool2d(1)
        self.pol_fc_sp = nn.Linear(channels, 3)

        # --- Value head ---
        self.val_conv1 = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.val_bn1   = nn.BatchNorm2d(1)
        self.val_fc1   = nn.Linear(self.N * self.N, 256)
        self.val_fc2   = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _convert_numpy_state(self, state):
        """
        Konwertuje stan numpy -> tensor (B, 3, N, N)
        state: np.ndarray (N, N) lub (B, N, N) z wartościami {-1, 0, 1}
        """
        if state.ndim == 2:
            # pojedyncza plansza
            me_plane    = (state ==  1).astype(np.float32)
            empty_plane = (state ==  0).astype(np.float32)
            opp_plane   = (state == -1).astype(np.float32)
            planes = np.stack([me_plane, empty_plane, opp_plane], axis=0)  # (3, N, N)
            x = torch.from_numpy(planes).unsqueeze(0)  # (1, 3, N, N)
        elif state.ndim == 3:
            # batch plansz
            me_plane    = (state ==  1).astype(np.float32)
            empty_plane = (state ==  0).astype(np.float32)
            opp_plane   = (state == -1).astype(np.float32)
            planes = np.stack([me_plane, empty_plane, opp_plane], axis=1)  # (B, 3, N, N)
            x = torch.from_numpy(planes)
        else:
            raise ValueError(f"Nieobsługiwany kształt stanu: {state.shape}")
        return x

    def forward(self, x):
        # Jeśli wejście jest numpy, konwertujemy
        if isinstance(x, np.ndarray):
            x = self._convert_numpy_state(x)

        # x: tensor (B, 3, N, N)
        x = x.contiguous(memory_format=torch.channels_last)

        h = self.stem(x)
        h = self.trunk(h)

        # policy head - pola
        p = self.pol_conv1(h)
        p = F.relu(self.pol_bn1(p), inplace=True)
        p = self.pol_conv2(p)
        p_board = p.flatten(1)  # (B, N*N)

        # policy head - specjalne akcje
        g = self.pol_gap(h).flatten(1)
        p_special = self.pol_fc_sp(g)  # (B, 3)

        policy_logits = torch.cat([p_board, p_special], dim=1)  # (B, N*N+3)

        # value head
        v = self.val_conv1(h)
        v = F.relu(self.val_bn1(v), inplace=True)
        v = v.flatten(1)  # (B, N*N)
        v = F.relu(self.val_fc1(v), inplace=True)
        v = torch.tanh(self.val_fc2(v)).squeeze(1)  # (B,)

        return {'policy_logits': policy_logits, 'value': v}
