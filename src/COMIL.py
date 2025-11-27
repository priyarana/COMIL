import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import initialize_weights



# ================================================================
#   GCT: Global Context Transform (Reusable Module)
#   From: GCT — Global Context Transform Block for Channel Attention
# ================================================================
class GCT(nn.Module):
    def __init__(self, channels, eps=1e-5):
        """
        channels: number of channel groups in input [N, C, D]
        eps: numerical stability
        """
        super().__init__()
        self.eps = eps

        # trainable scaling parameters
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        """
        x: [N, C, D]
        Applies GCT on channel dimension C.
        """
        # Channel L2-magnitude (Global Context)
        embedding = (x.pow(2).sum(2, keepdim=True) + self.eps).sqrt()
        embedding = embedding * self.alpha  # scale

        # Cross-channel normalization
        denom = (embedding.pow(2).mean(dim=1, keepdim=True) + self.eps).sqrt()
        norm = self.gamma / denom

        # Tanh gate
        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate            # [N, C, D]



# ================================================================
#   Attention Networks (unchanged)
# ================================================================
class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        layers = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(D, n_classes))
        self.attn = nn.Sequential(*layers)

    def forward(self, x):
        return self.attn(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        a = [nn.Linear(L, D), nn.Tanh()]
        b = [nn.Linear(L, D), nn.Sigmoid()]
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



# ================================================================
#                        COMIL MODEL
# ================================================================
class COMIL(nn.Module):

    def __init__(self,
                 gate=True,
                 size_arg="small",
                 dropout=True,
                 k_sample=8,
                 n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(),
                 subtyping=False,
                 epsilon=1e-5):

        super().__init__()

        size_dict = {
            "small": [2048, 1024, 512],
            "big":   [2048, 1024, 768]
        }

        in_dim, hid_dim, att_dim = size_dict[size_arg]

        self.n_classes = n_classes
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

        # ======================
        #  GCT Module (Reusable)
        # ======================
        self.gct = GCT(channels=7, eps=epsilon)

        # ======================
        #  Encoder + Attention
        # ======================
        layers = [
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim)
        ]
        if dropout:
            layers.append(nn.Dropout(0.25))

        if gate:
            att_module = Attn_Net_Gated(L=hid_dim, D=att_dim, dropout=dropout, n_classes=1)
        else:
            att_module = Attn_Net(L=hid_dim, D=att_dim, dropout=dropout, n_classes=1)

        layers.append(att_module)
        self.attention_net = nn.Sequential(*layers)

        # ======================
        #  Bag + Instance Classifiers
        # ======================
        self.classifier = nn.Linear(hid_dim, n_classes)

        self.instance_classifiers = nn.ModuleList([
            nn.Linear(hid_dim, 2) for _ in range(n_classes)
        ])

        initialize_weights(self)


    # -----------------------------------------------------------
    # Utility targets
    @staticmethod
    def pos_targets(k, device): return torch.ones(k, device=device, dtype=torch.long)
    @staticmethod
    def neg_targets(k, device): return torch.zeros(k, device=device, dtype=torch.long)
    # -----------------------------------------------------------


    # ----------------------------------------------------------------
    def inst_eval(self, A, h, classifier):
        """Top-k positive / Top-k negative selection"""
        A = A.view(1, -1)
        device = h.device
        N = A.size(1)

        if self.k_sample >= N:
            return torch.tensor(0., device=device), None, None

        # top-k positive
        pos_ids = torch.topk(A, self.k_sample, dim=1)[1][0]
        pos_feats = h[pos_ids]

        # top-k negative
        neg_ids = torch.topk(-A, self.k_sample, dim=1)[1][0]
        neg_feats = h[neg_ids]

        targets = torch.cat([
            self.pos_targets(self.k_sample, device),
            self.neg_targets(self.k_sample, device)
        ])
        feats = torch.cat([pos_feats, neg_feats])

        logits = classifier(feats)
        preds = logits.argmax(dim=1)
        loss = self.instance_loss_fn(logits, targets)

        return loss, preds, targets


    def inst_eval_out(self, A, h, classifier):
        """Subtyping negative-only sampling."""
        A = A.view(1, -1)
        device = h.device
        N = A.size(1)

        if self.k_sample >= N:
            return torch.tensor(0., device=device), None, None

        ids = torch.topk(A, self.k_sample, dim=1)[1][0]
        feats = h[ids]

        targets = self.neg_targets(self.k_sample, device)
        logits = classifier(feats)
        preds = logits.argmax(dim=1)
        loss = self.instance_loss_fn(logits, targets)

        return loss, preds, targets
    # ----------------------------------------------------------------


    # ================================================================
    #                           FORWARD
    # ================================================================
    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        """
        h: [N, 7, 2048]
        """
        device = h.device

        # -----------------------
        # 1) GCT (Reusable)
        # -----------------------
        h = self.gct(h)            # [N, 7, 2048]

        # -----------------------
        # 2) Channel pooling
        # -----------------------
        h = F.adaptive_avg_pool1d(h.permute(0, 2, 1), 1).squeeze(-1)
        # now h = [N, 2048]

        # -----------------------
        # 3) Attention network
        # -----------------------
        A, h = self.attention_net(h)
        A = A.transpose(1, 0)      # [1, N]

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)

        # -----------------------
        # 4) Instance-level loss (optional)
        # -----------------------
        results = {}
        if instance_eval:
            total_loss = 0.
            preds_all = []
            targets_all = []

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
                    preds_all.extend(preds.cpu().numpy())
                    targets_all.extend(t.cpu().numpy())

            if self.subtyping:
                total_loss /= len(self.instance_classifiers)

            results = {
                "instance_loss": total_loss,
                "inst_preds": np.array(preds_all),
                "inst_labels": np.array(targets_all)
            }

        # -----------------------
        # 5) MIL pooling
        # -----------------------
        M = A @ h                  # [1, 1024]

        # -----------------------
        # 6) Bag classification
        # -----------------------
        logits = self.classifier(M)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = logits.argmax(dim=1)

        if return_features:
            results["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results
