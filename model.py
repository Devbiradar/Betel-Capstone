"""
model.py
────────
Three quantum-classical hybrid architectures:

  Model 1 – QuantumTransferLearning   (Proposed)
  Model 2 – QuantumAttentionHybrid    (Main proposed model)
  Model 3 – EfficientNetFeatureExtractor + QuantumKernel  (for QSVM)

All PennyLane circuits use `default.qubit` with torch backprop
so they integrate seamlessly into the PyTorch autograd graph.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torchvision.models as models

# ─── Shared Helpers ──────────────────────────────────────────────────────────

N_QUBITS = 6   # number of qubits used across all models

def _make_device(n_qubits: int = N_QUBITS):
    return qml.device("default.qubit", wires=n_qubits)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Quantum Transfer Learning (Proposed)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_qtl_circuit(n_qubits: int, n_layers: int):
    """
    AngleEmbedding + BasicEntanglerLayers QNode.
    Returns a differentiable PennyLane QNode (torch interface).
    """
    dev = _make_device(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Encode classical features as rotation angles on Bloch sphere
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        # Parameterised entangling layers
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        # Measure expectation values of Pauli-Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class QuantumTransferLearning(nn.Module):
    """
    Model 1 — Quantum Transfer Learning (Proposed)
    ────────────────────────────────────────────────
    EfficientNetB0 backbone
        → feature reduction Dense (1280 → n_qubits) with BN + Dropout
        → PennyLane quantum layer (AngleEmbedding + BasicEntanglerLayers)
        → Dense classification head
        → Softmax (implicit in CrossEntropyLoss)
    """

    def __init__(
        self,
        n_classes:  int   = 6,
        n_qubits:   int   = N_QUBITS,
        n_layers:   int   = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.n_qubits = n_qubits

        # ── Backbone (EfficientNetB0, ImageNet pretrained) ───────────────────
        backbone         = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features    = backbone.features  # outputs [B, 1280, 7, 7]
        self.avgpool     = backbone.avgpool   # → [B, 1280, 1, 1]

        # ── Pre-quantum feature compression ──────────────────────────────────
        # 1280 → n_qubits, Tanh bounds values to [-1, 1] for AngleEmbedding
        self.pre_quantum = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_qubits),
            nn.Tanh(),  # scale to [-1, 1] ≈ angles in radians
        )

        # ── Quantum layer ────────────────────────────────────────────────────
        circuit         = _build_qtl_circuit(n_qubits, n_layers)
        weight_shapes   = {"weights": (n_layers, n_qubits)}
        self.qlayer     = qml.qnn.TorchLayer(circuit, weight_shapes)

        # ── Post-quantum classifier ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self._freeze_backbone()

    # ── Freeze / Unfreeze API  (called by train.py for 2-phase training) ──────
    def _freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                 # [B, 1280, 7, 7]
        x = self.avgpool(x)                  # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)              # [B, 1280]
        x = self.pre_quantum(x)              # [B, n_qubits]
        x = self.qlayer(x)                   # [B, n_qubits] — quantum output
        x = self.classifier(x)              # [B, n_classes]
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Quantum Attention Hybrid  (Main Proposed Model)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_quantum_attention_circuit(n_qubits: int, n_layers: int):
    """
    Variational circuit used as a quantum attention scorer.
    Architecture: AngleEmbedding(Y) → [RY + RZ + ring CNOT] × n_layers → <Z>.
    """
    dev = _make_device(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Amplitude encode the classical-attention-gated features
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        # Variational layers with local rotations + ring entanglement
        for layer_idx in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer_idx, q, 0], wires=q)
                qml.RZ(weights[layer_idx, q, 1], wires=q)
            # Ring entanglement: 0-1, 1-2, …, (n-1)-0
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


class QuantumAttentionHybrid(nn.Module):
    """
    Model 2 — Quantum Attention Hybrid  (Main Proposed Model)
    ──────────────────────────────────────────────────────────
    EfficientNetB0
        → Classical sigmoid attention gate (soft channel attention)
        → Quantum variational circuit scores the attended features
        → Fuse [classical-attended, quantum-scores] → Dense head
        → Softmax (implicit in CrossEntropyLoss)
    """

    def __init__(
        self,
        n_classes:   int   = 6,
        n_qubits:    int   = N_QUBITS,
        n_layers:    int   = 3,
        dropout:     float = 0.3,
        feat_dim:    int   = 1280,
    ):
        super().__init__()
        self.n_qubits  = n_qubits
        self.feat_dim  = feat_dim

        # ── Backbone ─────────────────────────────────────────────────────────
        backbone      = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # ── Classical channel-attention gate ─────────────────────────────────
        # SE-style: squeeze → excite → sigmoid gating
        self.classical_attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.BatchNorm1d(feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, feat_dim),
            nn.Sigmoid(),   # attention weights ∈ (0, 1)
        )

        # ── Feature compression for quantum layer ─────────────────────────────
        # Compress attended features into n_qubits values ∈ [-1, 1]
        self.compress = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_qubits),
            nn.Tanh(),
        )

        # ── Quantum attention circuit ─────────────────────────────────────────
        circuit       = _build_quantum_attention_circuit(n_qubits, n_layers)
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}   # RY + RZ per qubit
        self.q_attn   = qml.qnn.TorchLayer(circuit, weight_shapes)

        # ── Fusion: concatenate classical features + quantum scores ───────────
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + n_qubits, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Extract backbone features
        feat = self.features(x)          # [B, 1280, 7, 7]
        feat = self.avgpool(feat)        # [B, 1280, 1, 1]
        feat = torch.flatten(feat, 1)   # [B, 1280]

        # 2. Classical attention gate
        attn   = self.classical_attn(feat)   # [B, 1280] — sigmoid weights
        gated  = feat * attn                 # element-wise gating

        # 3. Compress gated features → quantum layer input
        q_in  = self.compress(gated)         # [B, n_qubits]

        # 4. Quantum attention scoring
        q_out = self.q_attn(q_in)            # [B, n_qubits]

        # 5. Fuse classical-gated + quantum
        fused = torch.cat([gated, q_out], dim=1)   # [B, 1280 + n_qubits]
        fused = self.fusion(fused)                  # [B, 512]

        # 6. Classify
        return self.classifier(fused)                # [B, n_classes]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — EfficientNet Feature Extractor  (for QSVM)
# ═══════════════════════════════════════════════════════════════════════════════

class EfficientNetExtractor(nn.Module):
    """
    Frozen EfficientNetB0 used purely for feature extraction in the QSVM pipeline.
    Outputs 1280-dim vectors (no classification head).
    """

    def __init__(self):
        super().__init__()
        backbone      = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # [B, 1280]


# ─── Quantum Kernel (ZZFeatureMap-style) ─────────────────────────────────────

class QuantumKernel:
    """
    ZZFeatureMap-style quantum kernel for use with sklearn SVC(kernel='precomputed').

    Steps
    ─────
    1. Apply ZZ feature map to x1   (Hadamard + RZ + ZZ two-qubit phase).
    2. Apply ADJOINT of ZZ map for x2.
    3. Measure |⟨ψ(x1)|ψ(x2)⟩|² as the probability of measuring |0…0⟩.

    Parameters
    ----------
    n_qubits : int  – must equal the PCA-reduced feature dimension.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_reps: int = 2):
        self.n_qubits = n_qubits
        self.n_reps   = n_reps
        self._dev     = qml.device("default.qubit", wires=n_qubits)
        self._build()

    # ── ZZ Feature Map ────────────────────────────────────────────────────────
    @staticmethod
    def _zz_layer(x, wires):
        for i, w in enumerate(wires):
            qml.Hadamard(wires=w)
            qml.RZ(2.0 * x[i], wires=w)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            # ZZ interaction angle: 2(π − x_i)(π − x_{i+1})
            angle = 2.0 * (np.pi - x[i]) * (np.pi - x[i + 1])
            qml.RZ(angle, wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    def _build(self):
        dev     = self._dev
        n       = self.n_qubits
        n_reps  = self.n_reps
        zz      = self._zz_layer

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            # Apply ZZ feature map for x1  |ψ(x1)⟩
            for _ in range(n_reps):
                zz(x1, list(range(n)))
            # Apply adjoint ZZ feature map for x2  ⟨ψ(x2)|
            for _ in range(n_reps):
                qml.adjoint(zz)(x2, list(range(n)))
            return qml.probs(wires=range(n))

        self._circuit = kernel_circuit

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Kernel value k(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|²."""
        probs = self._circuit(x1, x2)
        return float(probs[0])

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute a full kernel matrix K[i, j] = k(X1[i], X2[j]).
        Includes a tqdm progress bar and halves computation time for square matrices.
        """
        from tqdm import tqdm
        N, M = len(X1), len(X2)
        K = np.zeros((N, M))

        is_symmetric = (N == M) and np.array_equal(X1, X2)
        total_steps = (N * (N + 1)) // 2 if is_symmetric else N * M

        with tqdm(total=total_steps, desc="  Kernel Progress", leave=False) as pbar:
            for i in range(N):
                start_j = i if is_symmetric else 0
                for j in range(start_j, M):
                    if is_symmetric and i == j:
                        K[i, j] = 1.0
                    else:
                        val = self.__call__(X1[i], X2[j])
                        K[i, j] = val
                        if is_symmetric:
                            K[j, i] = val
                    pbar.update(1)
        return K
