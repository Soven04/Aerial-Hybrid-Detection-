"""
Hybrid Aerial Object Detection Model
=====================================
Architecture: Sparse RCNN + Transformer (AerialDet-Hybrid)

Pipeline:
  Input Image
      │
  ResNet-50 + FPN Backbone
      │
  Deformable Transformer Encoder   ← Captures long-range dependencies
      │
  Learnable Proposal Boxes (N=100)
      │
  Sparse RCNN Dynamic Head         ← Iterative refinement (6 stages)
      │
  Final Predictions (boxes + labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────
#  1. Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for feature maps.
    Produces encodings of shape (B, C, H, W).
    """
    def __init__(self, d_model: int, max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.d_model = d_model

        # Build fixed 2D sin/cos encoding
        pe = torch.zeros(d_model, max_h, max_w)
        d_half = d_model // 2

        # Height dimension
        pos_h = torch.arange(max_h).unsqueeze(1).float()      # (H, 1)
        div   = torch.exp(torch.arange(0, d_half, 2).float()
                          * -(math.log(10000.0) / d_half))
        pe[0:d_half:2,   :, 0] = torch.sin(pos_h * div).T
        pe[1:d_half:2,   :, 0] = torch.cos(pos_h * div).T

        # Width dimension
        pos_w = torch.arange(max_w).unsqueeze(1).float()      # (W, 1)
        pe[d_half::2,    0, :] = torch.sin(pos_w * div).T
        pe[d_half+1::2,  0, :] = torch.cos(pos_w * div).T

        self.register_buffer('pe', pe.unsqueeze(0))            # (1, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        return x + self.pe[:, :C, :H, :W]


# ─────────────────────────────────────────────
#  2. Deformable Attention (simplified)
# ─────────────────────────────────────────────
class DeformableAttention(nn.Module):
    """
    Simplified multi-scale deformable attention.
    Full version samples from multiple feature levels;
    here we do single-scale for clarity with the same interface.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # Offset network: predicts (x, y) offsets per head per point
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj  = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias,   0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias,  0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, query: torch.Tensor,
                reference_points: torch.Tensor,
                input_flatten: torch.Tensor,
                spatial_shapes: torch.Tensor) -> torch.Tensor:
        """
        query:            (B, N_q, C)
        reference_points: (B, N_q, 2)  normalized [0,1]
        input_flatten:    (B, N_kv, C)
        spatial_shapes:   (n_levels, 2) — unused in simplified version
        """
        B, N_q, C = query.shape
        N_kv = input_flatten.shape[1]

        values   = self.value_proj(input_flatten)                 # (B, N_kv, C)
        offsets  = self.sampling_offsets(query)                   # (B, N_q, H*P*2)
        offsets  = offsets.view(B, N_q, self.n_heads, self.n_points, 2)
        offsets  = torch.tanh(offsets) * 0.5                     # clamp to [-0.5, 0.5]

        attn_w   = self.attention_weights(query)                  # (B, N_q, H*P)
        attn_w   = attn_w.view(B, N_q, self.n_heads, self.n_points)
        attn_w   = F.softmax(attn_w, dim=-1)

        # Simple gather: use standard attention as fallback
        # (In production replace with grid_sample over multi-scale features)
        attn_scores = torch.einsum('bqc,bkc->bqk', query, values) / math.sqrt(C)
        attn_scores = F.softmax(attn_scores, dim=-1)              # (B, N_q, N_kv)
        out = torch.einsum('bqk,bkc->bqc', attn_scores, values)
        return self.output_proj(out)                              # (B, N_q, C)


# ─────────────────────────────────────────────
#  3. Transformer Encoder Layer
# ─────────────────────────────────────────────
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = DeformableAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                reference_points: torch.Tensor,
                spatial_shapes: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual
        src2 = self.self_attn(src, reference_points, src, spatial_shapes)
        src  = self.norm1(src + self.dropout(src2))
        # FFN + residual
        src  = self.norm2(src + self.dropout(self.ffn(src)))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor,
                reference_points: torch.Tensor,
                spatial_shapes: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, reference_points, spatial_shapes)
        return src


# ─────────────────────────────────────────────
#  4. FPN Neck
# ─────────────────────────────────────────────
class FPN(nn.Module):
    """
    Feature Pyramid Network.
    Takes C3, C4, C5 from backbone → P3, P4, P5, P6.
    """
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])
        self.extra_conv = nn.Conv2d(in_channels[-1], out_channels, 3, stride=2, padding=1)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features: [C3, C4, C5]
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i],
                                            size=laterals[i-1].shape[-2:],
                                            mode='nearest')
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        outs.append(self.extra_conv(features[-1]))   # P6
        return outs   # [P3, P4, P5, P6]


# ─────────────────────────────────────────────
#  5. Sparse RCNN Dynamic Head
# ─────────────────────────────────────────────
class DynamicConv(nn.Module):
    """
    Dynamic convolution: proposal features condition the instance kernels.
    Core innovation of Sparse RCNN.
    """
    def __init__(self, d_model: int = 256, roi_size: int = 7,
                 dynamic_dim: int = 64, dynamic_num: int = 2):
        super().__init__()
        self.d_model    = d_model
        self.roi_size   = roi_size
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        roi_feat_dim    = d_model * roi_size * roi_size

        # Generate dynamic parameters from proposal feature
        self.param_linear = nn.Linear(d_model, dynamic_dim * d_model
                                       + dynamic_dim * dynamic_dim * (dynamic_num - 1)
                                       + dynamic_dim * d_model)
        self.norm1 = nn.LayerNorm(dynamic_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.act   = nn.ReLU(inplace=True)

        self.fc_layer   = nn.Linear(d_model * roi_size * roi_size, d_model)
        self.fc_norm    = nn.LayerNorm(d_model)

    def forward(self, roi_feat: torch.Tensor,
                pro_feat: torch.Tensor) -> torch.Tensor:
        """
        roi_feat:  (B*N, C, roi_size, roi_size)
        pro_feat:  (B*N, C)
        Returns:   (B*N, C)
        """
        BN = roi_feat.shape[0]

        # Flatten roi feature
        roi_flat = roi_feat.view(BN, self.d_model, -1)   # (BN, C, r²)
        roi_flat = roi_flat.permute(0, 2, 1)              # (BN, r², C)

        # Generate instance-specific parameters
        params = self.param_linear(pro_feat)              # (BN, total_params)

        split1 = self.dynamic_dim * self.d_model
        split2 = self.dynamic_dim * self.dynamic_dim
        W1 = params[:, :split1].view(BN, self.d_model, self.dynamic_dim)
        W2 = params[:, split1:split1+split2].view(BN, self.dynamic_dim, self.dynamic_dim)
        W3 = params[:, split1+split2:].view(BN, self.dynamic_dim, self.d_model)

        # Apply dynamic kernels
        x = torch.bmm(roi_flat, W1)                      # (BN, r², dyn_dim)
        x = self.act(self.norm1(x.mean(1)))               # (BN, dyn_dim)
        x = x.unsqueeze(1)
        x = torch.bmm(x, W2).squeeze(1)                  # (BN, dyn_dim)
        x = self.act(x)
        x = x.unsqueeze(1)
        x = torch.bmm(x, W3).squeeze(1)                  # (BN, C)
        return self.norm2(x)


class SparseSingleStage(nn.Module):
    """
    One stage of Sparse RCNN iterative refinement head.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 num_cls: int = 15, roi_size: int = 7,
                 dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Self-attention among proposals
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Dynamic convolution
        self.dynamic_conv = DynamicConv(d_model, roi_size)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Prediction heads
        self.cls_head = nn.Linear(d_model, num_cls)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),   # (dx, dy, dw, dh)
        )

    def forward(self, roi_feat: torch.Tensor,
                pro_feat: torch.Tensor,
                proposal_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        roi_feat:       (B, N, C, r, r)
        pro_feat:       (B, N, C)
        proposal_boxes: (B, N, 4)  cx,cy,w,h normalized
        Returns: cls_logits (B,N,num_cls), box_deltas (B,N,4), updated pro_feat (B,N,C)
        """
        B, N, C, r, _ = roi_feat.shape

        # 1. Self-attention among proposals
        q = k = v = pro_feat
        attn_out, _ = self.self_attn(q, k, v)
        pro_feat = self.norm1(pro_feat + self.dropout(attn_out))

        # 2. Dynamic conv (proposal-conditioned RoI interaction)
        roi_flat = roi_feat.view(B * N, C, r, r)
        pro_flat = pro_feat.view(B * N, C)
        dyn_out  = self.dynamic_conv(roi_flat, pro_flat)   # (B*N, C)
        pro_feat = self.norm2(pro_feat + self.dropout(dyn_out.view(B, N, C)))

        # 3. FFN
        pro_feat = self.norm3(pro_feat + self.dropout(self.ffn(pro_feat)))

        # 4. Predict
        cls_logits = self.cls_head(pro_feat)   # (B, N, num_cls)
        box_deltas = self.reg_head(pro_feat)   # (B, N, 4)

        return cls_logits, box_deltas, pro_feat


class SparseRCNNHead(nn.Module):
    """
    Full iterative Sparse RCNN head with num_stages stages.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 num_cls: int = 15, num_proposals: int = 100,
                 num_stages: int = 6, roi_size: int = 7):
        super().__init__()
        self.num_stages = num_stages
        self.roi_size   = roi_size

        self.stages = nn.ModuleList([
            SparseSingleStage(d_model, n_heads, num_cls, roi_size)
            for _ in range(num_stages)
        ])

    def _roi_align(self, feature_map: torch.Tensor,
                   boxes: torch.Tensor) -> torch.Tensor:
        """
        Simplified RoI extraction via grid_sample.
        feature_map: (B, C, H, W)
        boxes:       (B, N, 4)  cx,cy,w,h in [0,1]
        Returns:     (B, N, C, roi_size, roi_size)
        """
        B, N, _ = boxes.shape
        cx, cy, w, h = boxes.unbind(-1)                  # each (B, N)

        x0 = (cx - w / 2).clamp(0, 1)
        y0 = (cy - h / 2).clamp(0, 1)
        x1 = (cx + w / 2).clamp(0, 1)
        y1 = (cy + h / 2).clamp(0, 1)

        r = self.roi_size
        roi_feats = []
        for b in range(B):
            feats_b = []
            for n in range(N):
                # Build sampling grid for this box
                xs = torch.linspace(x0[b, n].item(), x1[b, n].item(), r,
                                    device=boxes.device)
                ys = torch.linspace(y0[b, n].item(), y1[b, n].item(), r,
                                    device=boxes.device)
                grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
                grid = torch.stack([grid_x * 2 - 1, grid_y * 2 - 1], dim=-1)
                grid = grid.unsqueeze(0)                 # (1, r, r, 2)

                feat = F.grid_sample(feature_map[b:b+1], grid,
                                     align_corners=True, mode='bilinear')  # (1, C, r, r)
                feats_b.append(feat.squeeze(0))          # (C, r, r)
            roi_feats.append(torch.stack(feats_b, 0))   # (N, C, r, r)

        return torch.stack(roi_feats, 0)                 # (B, N, C, r, r)

    def forward(self, feature_map: torch.Tensor,
                proposal_boxes: torch.Tensor,
                proposal_feats: torch.Tensor) -> List[Dict]:
        """
        feature_map:    (B, C, H, W)
        proposal_boxes: (B, N, 4)  learnable, cx,cy,w,h normalized
        proposal_feats: (B, N, C)  learnable
        Returns list of {cls_logits, boxes} per stage.
        """
        all_outputs = []
        boxes  = proposal_boxes.clone()
        feats  = proposal_feats.clone()

        for stage in self.stages:
            roi_feat = self._roi_align(feature_map, boxes)      # (B,N,C,r,r)
            cls_logits, box_deltas, feats = stage(roi_feat, feats, boxes)

            # Apply deltas  (simple additive refinement)
            boxes = boxes + 0.1 * box_deltas.tanh()
            boxes = boxes.clamp(0, 1)

            all_outputs.append({'cls_logits': cls_logits, 'boxes': boxes})

        return all_outputs


# ─────────────────────────────────────────────
#  6. ResNet-like Backbone (lightweight)
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class LightBackbone(nn.Module):
    """
    Lightweight backbone producing C3, C4, C5 feature maps.
    Replace with torchvision ResNet-50 for production use.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Stage 2 → C2 (stride 4)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128),
        )
        # Stage 3 → C3 (stride 8)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ResBlock(256), ResBlock(256),
        )
        # Stage 4 → C4 (stride 16)
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            ResBlock(512), ResBlock(512), ResBlock(512),
        )
        # Stage 5 → C5 (stride 32)
        self.stage5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            ResBlock(1024), ResBlock(1024),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x  = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c3, c4, c5]       # strides 8, 16, 32


# ─────────────────────────────────────────────
#  7. AerialDet-Hybrid (Full Model)
# ─────────────────────────────────────────────
class AerialDetHybrid(nn.Module):
    """
    Hybrid Aerial Object Detection Model
    =====================================
    Backbone  : Lightweight ResNet (C3/C4/C5)
    Neck      : FPN (P3–P6)
    Encoder   : Deformable Transformer Encoder (global context)
    Head      : Sparse RCNN Dynamic Head (iterative refinement × 6)

    Particularly suited for aerial/drone imagery where:
    - Objects are small and densely packed
    - Scale variation is extreme
    - Rotation/orientation matters
    - Long-range context helps discriminate similar-looking objects
    """
    def __init__(self,
                 num_classes:    int = 15,
                 num_proposals:  int = 100,
                 d_model:        int = 256,
                 n_heads:        int = 8,
                 num_enc_layers: int = 6,
                 num_stages:     int = 6,
                 roi_size:       int = 7,
                 dropout:        float = 0.1,
                 in_channels:    int = 3):
        super().__init__()
        self.num_proposals = num_proposals
        self.d_model       = d_model
        self.num_classes   = num_classes

        # ── Backbone ──────────────────────────────
        self.backbone = LightBackbone(in_channels)

        # ── FPN Neck ──────────────────────────────
        self.fpn = FPN([256, 512, 1024], d_model)

        # ── Positional Encoding ───────────────────
        self.pos_enc = PositionalEncoding2D(d_model)

        # ── Transformer Encoder ───────────────────
        self.encoder = TransformerEncoder(
            d_model, n_heads, num_enc_layers,
            dim_feedforward=1024, dropout=dropout
        )

        # ── Sparse RCNN Head ──────────────────────
        self.sparse_rcnn = SparseRCNNHead(
            d_model, n_heads, num_classes,
            num_proposals, num_stages, roi_size
        )

        # ── Learnable Proposals ───────────────────
        # proposal_boxes: normalized (cx, cy, w, h) in [0, 1]
        self.proposal_boxes = nn.Embedding(num_proposals, 4)
        self.proposal_feats = nn.Embedding(num_proposals, d_model)

        self._init_proposals()
        self._init_weights()

    def _init_proposals(self):
        nn.init.uniform_(self.proposal_boxes.weight[:, :2], 0.1, 0.9)  # cx, cy
        nn.init.uniform_(self.proposal_boxes.weight[:, 2:], 0.05, 0.3) # w, h
        nn.init.normal_(self.proposal_feats.weight, 0, 0.01)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _flatten_features(self, fpn_feats: List[torch.Tensor]
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flatten all FPN levels into a single token sequence.
        Returns:
          flat_src:    (B, sum(H_i*W_i), C)
          ref_points:  (B, sum(H_i*W_i), 2)  normalized centers
        """
        B = fpn_feats[0].shape[0]
        tokens, refs = [], []
        for feat in fpn_feats:
            _, C, H, W = feat.shape
            feat = self.pos_enc(feat)
            flat = feat.flatten(2).permute(0, 2, 1)        # (B, H*W, C)
            tokens.append(flat)
            # Generate reference point grid
            ys = torch.linspace(0.5/H, 1 - 0.5/H, H, device=feat.device)
            xs = torch.linspace(0.5/W, 1 - 0.5/W, W, device=feat.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            ref = torch.stack([grid_x, grid_y], -1).view(-1, 2)
            refs.append(ref.unsqueeze(0).expand(B, -1, -1))

        flat_src   = torch.cat(tokens, dim=1)              # (B, N_total, C)
        ref_points = torch.cat(refs,   dim=1)              # (B, N_total, 2)
        return flat_src, ref_points

    def forward(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        images: (B, 3, H, W)  normalized to [0, 1]

        Returns list of dicts, one per Sparse RCNN stage:
          {
            'cls_logits': (B, N_proposals, num_classes),
            'boxes':      (B, N_proposals, 4)  cx,cy,w,h in [0,1]
          }
        Only the LAST dict is used at inference; all stages used during training.
        """
        B = images.shape[0]

        # ── 1. Backbone ──────────────────────────
        c_feats = self.backbone(images)                    # C3, C4, C5

        # ── 2. FPN ───────────────────────────────
        fpn_feats = self.fpn(c_feats)                      # P3, P4, P5, P6

        # ── 3. Transformer Encoder ────────────────
        flat_src, ref_points = self._flatten_features(fpn_feats)
        spatial_shapes = torch.tensor(
            [[f.shape[2], f.shape[3]] for f in fpn_feats],
            device=images.device
        )
        encoded = self.encoder(flat_src, ref_points, spatial_shapes)  # (B, N_tok, C)

        # Reconstruct main feature map from P3 tokens (largest resolution)
        P3_H, P3_W = fpn_feats[0].shape[2], fpn_feats[0].shape[3]
        n_p3 = P3_H * P3_W
        enc_p3 = encoded[:, :n_p3, :]                     # (B, H3*W3, C)
        enc_map = enc_p3.permute(0, 2, 1).view(B, self.d_model, P3_H, P3_W)

        # ── 4. Sparse RCNN Head ───────────────────
        boxes = self.proposal_boxes.weight.unsqueeze(0).expand(B, -1, -1)
        feats = self.proposal_feats.weight.unsqueeze(0).expand(B, -1, -1)
        # Sigmoid to keep boxes in [0, 1]
        boxes = torch.sigmoid(boxes)

        outputs = self.sparse_rcnn(enc_map, boxes, feats)

        return outputs

    @torch.no_grad()
    def predict(self, images: torch.Tensor,
                score_threshold: float = 0.5,
                nms_threshold:   float = 0.5) -> List[Dict]:
        """
        Inference helper. Returns final-stage detections per image.
        """
        self.eval()
        stage_outputs = self(images)
        final = stage_outputs[-1]   # last stage = best predictions

        B = images.shape[0]
        results = []
        for b in range(B):
            logits = final['cls_logits'][b]   # (N, num_cls)
            boxes  = final['boxes'][b]         # (N, 4)

            scores, labels = logits.softmax(-1).max(-1)
            keep = scores > score_threshold

            results.append({
                'scores': scores[keep],
                'labels': labels[keep],
                'boxes':  boxes[keep],   # cx, cy, w, h in [0, 1]
            })
        return results


# ─────────────────────────────────────────────
#  8. Loss Function
# ─────────────────────────────────────────────
class AerialDetLoss(nn.Module):
    """
    Combined loss for all Sparse RCNN stages.
    Uses Hungarian matching (bipartite) per stage.
    """
    def __init__(self, num_classes: int = 15,
                 cls_weight: float = 2.0, l1_weight: float = 5.0,
                 giou_weight: float = 2.0):
        super().__init__()
        self.num_classes  = num_classes
        self.cls_weight   = cls_weight
        self.l1_weight    = l1_weight
        self.giou_weight  = giou_weight

    def giou_loss(self, pred_boxes: torch.Tensor,
                  gt_boxes: torch.Tensor) -> torch.Tensor:
        """GIoU loss for boxes in cx,cy,w,h format."""
        # Convert to x1y1x2y2
        def to_xyxy(b):
            return torch.stack([b[..., 0] - b[..., 2]/2,
                                 b[..., 1] - b[..., 3]/2,
                                 b[..., 0] + b[..., 2]/2,
                                 b[..., 1] + b[..., 3]/2], dim=-1)
        p = to_xyxy(pred_boxes)
        g = to_xyxy(gt_boxes)

        inter_x1 = torch.max(p[..., 0], g[..., 0])
        inter_y1 = torch.max(p[..., 1], g[..., 1])
        inter_x2 = torch.min(p[..., 2], g[..., 2])
        inter_y2 = torch.min(p[..., 3], g[..., 3])
        inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
        area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
        union  = area_p + area_g - inter + 1e-6
        iou    = inter / union

        encl_x1 = torch.min(p[..., 0], g[..., 0])
        encl_y1 = torch.min(p[..., 1], g[..., 1])
        encl_x2 = torch.max(p[..., 2], g[..., 2])
        encl_y2 = torch.max(p[..., 3], g[..., 3])
        encl    = (encl_x2 - encl_x1).clamp(0) * (encl_y2 - encl_y1).clamp(0) + 1e-6

        giou = iou - (encl - union) / encl
        return 1 - giou

    def forward(self, stage_outputs: List[Dict],
                targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        stage_outputs: list of {'cls_logits': (B,N,C), 'boxes': (B,N,4)} per stage
        targets:       list of {'labels': (M,), 'boxes': (M,4)} per image
        """
        total_cls  = torch.tensor(0., requires_grad=True)
        total_l1   = torch.tensor(0., requires_grad=True)
        total_giou = torch.tensor(0., requires_grad=True)
        n_stages   = len(stage_outputs)

        for stage_out in stage_outputs:
            cls_logits = stage_out['cls_logits']   # (B, N, num_cls)
            boxes      = stage_out['boxes']         # (B, N, 4)
            B, N, _    = cls_logits.shape

            for b, tgt in enumerate(targets):
                gt_boxes  = tgt['boxes']   # (M, 4)
                gt_labels = tgt['labels']  # (M,)
                M = gt_boxes.shape[0]
                if M == 0:
                    continue

                # Simple nearest-proposal matching (replace with Hungarian for production)
                pred_boxes_b = boxes[b]     # (N, 4)
                cost = torch.cdist(pred_boxes_b, gt_boxes, p=1)  # (N, M)
                matched_gt = cost.argmin(dim=1)[:M]              # (M,)
                matched_pred = torch.arange(M, device=boxes.device)

                # Classification loss (focal-style BCE for matched)
                matched_logits = cls_logits[b][matched_pred]     # (M, num_cls)
                matched_labels = gt_labels                        # (M,)
                cls_tgt = torch.zeros_like(matched_logits)
                cls_tgt.scatter_(1, matched_labels.unsqueeze(1), 1.)
                loss_cls = F.binary_cross_entropy_with_logits(matched_logits, cls_tgt)

                # L1 regression loss
                matched_boxes = pred_boxes_b[matched_pred]       # (M, 4)
                matched_gt_boxes = gt_boxes[matched_gt]          # (M, 4)
                loss_l1 = F.l1_loss(matched_boxes, matched_gt_boxes)

                # GIoU loss
                loss_giou = self.giou_loss(matched_boxes, matched_gt_boxes).mean()

                total_cls  = total_cls  + loss_cls
                total_l1   = total_l1   + loss_l1
                total_giou = total_giou + loss_giou

        scale = 1.0 / (n_stages * len(targets) + 1e-6)
        total_loss = (self.cls_weight  * total_cls  +
                      self.l1_weight   * total_l1   +
                      self.giou_weight * total_giou ) * scale

        return {
            'total':    total_loss,
            'cls':      total_cls  * scale,
            'l1':       total_l1   * scale,
            'giou':     total_giou * scale,
        }
