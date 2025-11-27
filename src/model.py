import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import initialize_weights


# -------------------------
#   ATTENTION NETWORKS
# -------------------------
class Attn_Net(nn.Module):
    """2-layer attention network"""
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        layers = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(D, n_classes))
        self.attention = nn.Sequential(*layers)

    def forward(self, x):
        A = self.attention(x)
        return A, x


class Attn_Net_Gated(nn.Module):
    """Gated attention network (TanH × Sigmoid)"""
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        a = [nn.Linear(L, D), nn.Tanh()]
        b = [nn.Linear(L, D), nn.sigmoid()]
        if dropout:
            a.append(nn.Dropout(0.5))
            b.append(nn.Dropout(0.5))
        self.att_a = nn.Sequential(*a)
        self.att_b = nn.Sequential(*b)
        self.att_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.att_a(x)
        b = self.att_b(x)
        A = self.att_c(a * b)
        return A, x


# -------------------------
#   COMIL MODEL
# -------------------------
class COMIL(nn.Module):

    def __init__(self, gate=True, size_arg="small",
                 dropout=True, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(),
                 subtyping=False,
                 epsilon=1e-5):

        super().__init__()

        # Input feature = 2048-dim, typical for ResNet101
        size_dict = {
            "small": [2048, 1024, 512],
            "big":   [2048, 1024, 768]
        }
        in_dim, hid_dim, att_dim = size_dict[size_arg]

        self.epsilon = epsilon
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

        # --- Channel gating parameters ---
        self.alpha = nn.Parameter(torch.ones(1, 7, 1))
        self.gamma = nn.Parameter(torch.zeros(1, 7, 1))
        self.beta  = nn.Parameter(torch.zeros(1, 7, 1))

        # --- Feature backbone before attention ---
        fc_layers = [
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim),
        ]
        if dropout:
            fc_layers.append(nn.Dropout(0.25))

        # --- Attention network ---
        if gate:
            att_net = Attn_Net_Gated(L=hid_dim, D=att_dim, dropout=dropout, n_classes=1)
        else:
            att_net = Attn_Net(L=hid_dim, D=att_dim, dropout=dropout, n_classes=1)

        fc_layers.append(att_net)
        self.attention_net = nn.Sequential(*fc_layers)

        # --- Bag classifier ---
        self.classifier = nn.Linear(hid_dim, n_classes)

        # --- Instance-level classifiers (CLAM style) ---
        self.instance_classifiers = nn.ModuleList([
            nn.Linear(hid_dim, 2) for _ in range(n_classes)
        ])

        initialize_weights(self)

    # ----------------------------------------------------------------------
    # Utility target constructors
    @staticmethod
    def create_positive_targets(k, device):
        return torch.ones(k, device=device, dtype=torch.long)

    @staticmethod
    def create_negative_targets(k, device):
        return torch.zeros(k, device=device, dtype=torch.long)
    # ----------------------------------------------------------------------

    def inst_eval(self, A, h, classifier):
        """
        In-the-class instance loss: top-k highest attention = positive,
        top-k lowest attention = negative.
        """
        A = A.view(1, -1)               # [1, N]
        device = h.device
        N = A.size(1)

        if self.k_sample >= N:
            return torch.tensor(0., device=device), None, None

        # top-k high attention
        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1][0]
        top_p = h[top_p_ids]

        # top-k low attention
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][0]
        top_n = h[top_n_ids]

        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets])
        all_instances = torch.cat([top_p, top_n])

        logits = classifier(all_instances)
        preds = logits.argmax(dim=1)
        loss = self.instance_loss_fn(logits, all_targets)

        return loss, preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        """
        Out-of-class instance loss (for subtyping).
        Only top-k high attention → negative class.
        """
        A = A.view(1, -1)
        device = h.device
        N = A.size(1)

        if self.k_sample >= N:
            return torch.tensor(0., device=device), None, None

        top_ids = torch.topk(A, self.k_sample, dim=1)[1][0]
        top_instances = h[top_ids]

        targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_instances)
        preds = logits.argmax(dim=1)
        loss = self.instance_loss_fn(logits, targets)

        return loss, preds, targets

    # ----------------------------------------------------------------------
    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False):

        # h shape: [N, 7, 2048]
        device = h.device

        # ---------------------------
        # 1) CHANNEL GATING
        # ---------------------------
        embedding = (h.pow(2).sum(2, keepdim=True) + self.epsilon).sqrt() * self.alpha
        norm = self.gamma / ((embedding.pow(2).mean(1, keepdim=True) + self.epsilon).sqrt())
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        h = h * gate

        # Pool channels → [N, 2048]
        h = F.adaptive_avg_pool1d(h.permute(0, 2, 1), 1).squeeze(-1)

        # ---------------------------
        # 2) APPLY ATTENTION NETWORK
        # ---------------------------
        A, h = self.attention_net(h)     # A: [N,1], h: [N,1024]
        A = A.transpose(1, 0)            # → [1, N]

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)          # normalized attention

        # ---------------------------
        # 3) INSTANCE EVALUATION
        # ---------------------------
        results = {}
        if instance_eval:
            total_loss = 0.
            all_preds = []
            all_targets = []

            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

            for i, clf in enumerate(self.instance_classifiers):
                if inst_labels[i] == 1:
                    loss, preds, t = self.inst_eval(A, h, clf)
                elif self.subtyping:
                    loss, preds, t = self.inst_eval_out(A, h, clf)
                else:
                    continue

                total_loss += loss
                if preds is not None:
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(t.cpu().numpy())

            if self.subtyping:
                total_loss /= len(self.instance_classifiers)

            results["instance_loss"] = total_loss
            results["inst_preds"] = np.array(all_preds)
            results["inst_labels"] = np.array(all_targets)

        # ---------------------------
        # 4) BAG-LEVEL CLASSIFIER
        # ---------------------------
        M = A @ h                         # [1, 1024]
        logits = self.classifier(M)       # [1, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = logits.argmax(dim=1)

        if return_features:
            results["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results
