import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from einops import rearrange
from typing import Dict, List, Optional


class ConditionalCodePredictor(nn.Module):
    """Conditional code predictor using GPT-2 with pretrained weights.
    
    Args:
        d_model: Model dimension for input embeddings
        codebook_size: Size of the codebook vocabulary
        num_rq_layers: Number of residual quantization layers
        n_heads: Number of attention heads (only used if use_pretrained=False)
        n_layers: Number of transformer layers (only used if use_pretrained=False)
        dropout: Dropout rate
        use_pretrained: Whether to load pretrained GPT-2 weights
        pretrained_model: Pretrained model name ('gpt2', 'gpt2-medium', 'gpt2-large')
    """
    def __init__(self, d_model: int, codebook_size: int, num_rq_layers: int,
                 n_heads: int = 4, n_layers: int = 4, dropout: float = 0.1,
                 use_pretrained: bool = True, pretrained_model: str = 'gpt2',
                 freeze_gpt2: bool = True, unfreeze_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_rq_layers = num_rq_layers
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # Load pretrained GPT-2 model
            print(f"Loading pretrained GPT-2 model: {pretrained_model}")
            self.gpt2 = GPT2Model.from_pretrained(
                pretrained_model,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
            )
            # Get the hidden size of pretrained model (768 for gpt2, 1024 for gpt2-medium, etc.)
            self.gpt2_hidden_size = self.gpt2.config.n_embd
            
            # Freeze GPT-2 layers except the last `unfreeze_layers`
            if freeze_gpt2:
                for param in self.gpt2.parameters():
                    param.requires_grad = False
                # Unfreeze the last N transformer blocks
                total_blocks = len(self.gpt2.h)
                for i in range(max(0, total_blocks - unfreeze_layers), total_blocks):
                    for param in self.gpt2.h[i].parameters():
                        param.requires_grad = True
                # Always unfreeze layer norm
                for param in self.gpt2.ln_f.parameters():
                    param.requires_grad = True
                
                frozen = sum(1 for p in self.gpt2.parameters() if not p.requires_grad)
                unfrozen = sum(1 for p in self.gpt2.parameters() if p.requires_grad)
                print(f"  GPT-2 frozen params: {frozen}, unfrozen params: {unfrozen}")
            
            # Projection layers: d_model <-> gpt2_hidden_size
            self.input_proj = nn.Linear(d_model, self.gpt2_hidden_size)
            self.output_proj = nn.Linear(self.gpt2_hidden_size, d_model)
            
            # Layer embedding in GPT-2's hidden size
            self.layer_embed = nn.Embedding(num_rq_layers, self.gpt2_hidden_size)
        else:
            # Train from scratch with custom config
            print(f"Initializing GPT-2 from scratch: n_layers={n_layers}, n_heads={n_heads}")
            gpt_config = GPT2Config(
                vocab_size=codebook_size + 2,
                n_embd=d_model,
                n_head=n_heads,
                n_layer=n_layers,
                n_positions=512,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
            )
            self.gpt2 = GPT2Model(gpt_config)
            self.gpt2_hidden_size = d_model
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
            self.layer_embed = nn.Embedding(num_rq_layers, d_model)
        
        # Code prediction heads (always in d_model space)
        self.code_heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(num_rq_layers)
        ])
    
    def forward(self, code_embeds: List[torch.Tensor], condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = condition.size(0)
        condition = condition.unsqueeze(1) if condition.dim() == 2 else condition
        
        # Build a single sequence: [condition, layer1_emb, code1, layer2_emb, code2, ...]
        # This lets GPT-2 attend across all layers and build a richer representation
        seq_parts = []
        if self.use_pretrained:
            seq_parts.append(self.input_proj(condition))
        else:
            seq_parts.append(condition)
        
        for layer_idx, code_embed in enumerate(code_embeds):
            layer_emb = self.layer_embed(torch.tensor(layer_idx, device=condition.device))
            layer_emb = layer_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            seq_parts.append(layer_emb)
            
            if self.use_pretrained:
                ce = self.input_proj(code_embed.unsqueeze(1) if code_embed.dim() == 2 else code_embed)
            else:
                ce = code_embed.unsqueeze(1) if code_embed.dim() == 2 else code_embed
            seq_parts.append(ce)
        
        input_seq = torch.cat(seq_parts, dim=1)  # (B, 1 + 2*L, hidden)
        
        # Single forward through GPT-2
        gpt_out = self.gpt2(inputs_embeds=input_seq).last_hidden_state
        
        # Extract code prediction logits from positions after each code_embed
        # Positions: 0=condition, 1=layer0_emb, 2=code0, 3=layer1_emb, 4=code1, ...
        # Code prediction at positions 2, 4, 6, ... (after each code embed)
        all_logits = []
        gpt_features = []
        for layer_idx in range(len(code_embeds)):
            pos = 2 + layer_idx * 2  # position of code_embed for this layer
            if self.use_pretrained:
                output = self.output_proj(gpt_out[:, pos, :])
            else:
                output = gpt_out[:, pos, :]
            logits = self.code_heads[layer_idx](output)
            all_logits.append(logits)
            gpt_features.append(output)
        
        # Also extract a global representation from GPT-2 (last position)
        if self.use_pretrained:
            global_repr = self.output_proj(gpt_out[:, -1, :])
        else:
            global_repr = gpt_out[:, -1, :]
        
        return {
            'layer_logits': all_logits,
            'stacked_logits': torch.stack(all_logits, dim=1),
            'gpt_features': gpt_features,
            'global_repr': global_repr,
        }


class Stage3GPT2Predictor(nn.Module):
    """Stage 3 GPT-2 based predictor for classification and code prediction.
    
    Args:
        d_model: Model dimension
        codebook_size: Size of the codebook vocabulary
        num_rq_layers: Number of residual quantization layers
        num_classes: Number of classification classes
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout rate
        use_pretrained: Whether to load pretrained GPT-2 weights
        pretrained_model: Pretrained model name
    """
    def __init__(self, d_model: int, codebook_size: int = 512, num_rq_layers: int = 3,
                 num_classes: int = 10, n_heads: int = 4, n_layers: int = 4, dropout: float = 0.1,
                 use_pretrained: bool = True, pretrained_model: str = 'gpt2',
                 freeze_gpt2: bool = True, unfreeze_layers: int = 2,
                 bypass_gpt2_cls: bool = False):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_rq_layers = num_rq_layers
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.bypass_gpt2_cls = bypass_gpt2_cls
        
        self.class_embed = nn.Embedding(num_classes, d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)  # learnable CLS token for classification
        self.predictor = ConditionalCodePredictor(
            d_model, codebook_size, num_rq_layers, n_heads, n_layers, dropout,
            use_pretrained=use_pretrained, pretrained_model=pretrained_model,
            freeze_gpt2=freeze_gpt2, unfreeze_layers=unfreeze_layers
        )
        # Classifier input dimension depends on bypass mode
        if bypass_gpt2_cls:
            # Only use code embeddings (no GPT-2 global repr)
            cls_input_dim = d_model * num_rq_layers
        else:
            # Use both code embeddings and GPT-2 features
            cls_input_dim = d_model * num_rq_layers + d_model
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, code_embeds: List[torch.Tensor], labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = code_embeds[0].size(0)
        
        # Classification branch: always uses a learnable CLS token (no label leakage)
        cls_condition = self.cls_token.expand(B, -1)
        cls_output = self.predictor(code_embeds, cls_condition)
        code_combined = torch.cat([ce.squeeze(1) if ce.dim() == 3 else ce for ce in code_embeds], dim=-1)
        if self.bypass_gpt2_cls:
            cls_input = code_combined
        else:
            cls_input = torch.cat([code_combined, cls_output['global_repr']], dim=-1)
        cls_logits = self.classifier(cls_input)
        
        # Code prediction branch: uses label as condition (only during training)
        pred_output = None
        if labels is not None:
            condition = self.class_embed(labels)
            pred_output = self.predictor(code_embeds, condition)
        
        return {
            'layer_logits': pred_output['layer_logits'] if pred_output else cls_output['layer_logits'],
            'stacked_logits': pred_output['stacked_logits'] if pred_output else cls_output['stacked_logits'],
            'cls_logits': cls_logits
        }
    
    def get_loss(self, outputs: Dict, codes: List[torch.Tensor], labels: torch.Tensor,
                 cls_weight: float = 2.0, code_weights: List[float] = None) -> tuple:
        cls_loss = F.cross_entropy(outputs['cls_logits'], labels)
        if code_weights is None:
            code_weights = [1.0] * len(codes)
        pred_loss = sum(
            w * F.cross_entropy(logits, code) 
            for w, logits, code in zip(code_weights, outputs['layer_logits'], codes)
        ) / len(codes)
        total_loss = cls_weight * cls_loss + pred_loss
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'pred_loss': pred_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get the number of parameters in different parts of the model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gpt2_params = sum(p.numel() for p in self.predictor.gpt2.parameters())
        return {
            'total': total,
            'trainable': trainable,
            'gpt2': gpt2_params,
            'other': total - gpt2_params
        }
    
    @torch.no_grad()
    def generate(self, class_label: torch.Tensor, code_embeddings: nn.ModuleList) -> List[torch.Tensor]:
        B = class_label.size(0)
        condition = self.class_embed(class_label)
        generated_codes = []
        for layer_idx in range(self.num_rq_layers):
            if layer_idx == 0:
                dummy_embed = torch.zeros(B, self.d_model, device=class_label.device)
                code_embeds = [dummy_embed]
            else:
                code_embeds = [code_embeddings[i](generated_codes[i]) for i in range(layer_idx)]
            pred_output = self.predictor(code_embeds, condition)
            logits = pred_output['layer_logits'][min(layer_idx, len(pred_output['layer_logits']) - 1)]
            codes = logits.argmax(dim=-1)
            generated_codes.append(codes)
        return generated_codes
