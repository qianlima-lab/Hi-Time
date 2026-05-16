import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict

from configs import Config
from models.encoder import Stage1Encoder
from models.discretizer import Stage2Discretizer


def train_stage2(config: Config, encoder: Stage1Encoder, train_loader: DataLoader,
                 val_loader: DataLoader = None, device: torch.device = None) -> Stage2Discretizer:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    
    discretizer = Stage2Discretizer(
        d_model=config.model.d_model,
        codebook_size=config.model.codebook_size,
        num_rq_layers=config.model.num_quantize_layers,
        num_classes=config.data.num_classes
    ).to(device)
    
    # Pre-compute all encoder representations and init codebooks from full data
    all_reprs = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            all_reprs.append(encoder.encode(x))
    all_reprs_cat = torch.cat(all_reprs, dim=0)
    
    # Initialize each quantizer's codebook from the full representation set
    residual = all_reprs_cat
    for i, quantizer in enumerate(discretizer.residual_quantizer.quantizers):
        quantizer.init_codebook(residual.detach())
        # Also sync code_embedding
        code_emb = discretizer.residual_quantizer.code_embeddings[i]
        with torch.no_grad():
            code_emb.weight.data.copy_(quantizer.codebook.data)
        # Compute residual for next layer init
        dist = torch.cdist(residual, quantizer.codebook)
        indices = dist.argmin(dim=-1)
        quantized = torch.nn.functional.embedding(indices, quantizer.codebook)
        residual = residual - quantized
    
    n_codes = config.model.codebook_size
    for i, q in enumerate(discretizer.residual_quantizer.quantizers):
        used = len(torch.unique(torch.cdist(all_reprs_cat if i == 0 else residual, q.codebook).argmin(dim=-1)))
        print(f"  Codebook {i} init: {used}/{n_codes} codes used")
    
    optimizer = torch.optim.AdamW(
        discretizer.parameters(),
        lr=config.training.stage2_lr,
        weight_decay=config.training.stage2_weight_decay
    )
    
    scheduler = None
    stage2_use_sched = getattr(config.training, 'stage2_use_scheduler', config.training.use_scheduler)
    if stage2_use_sched:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.stage2_epochs,
            eta_min=config.training.scheduler_eta_min
        )
    
    for epoch in range(config.training.stage2_epochs):
        discretizer.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                repr = encoder.encode(x)
            
            optimizer.zero_grad()
            outputs = discretizer(repr)
            loss, _ = discretizer.get_loss(outputs, repr, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discretizer.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_correct += (outputs['logits'].argmax(dim=-1) == y).sum().item()
            total_samples += x.size(0)
        
        if scheduler is not None:
            scheduler.step()
        
        log_interval = max(1, config.training.stage2_epochs // 10)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            acc = total_correct / total_samples
            print(f"Stage2 Epoch {epoch + 1}/{config.training.stage2_epochs} | loss: {total_loss / total_samples:.4f} | acc: {acc:.4f}")
    
    return discretizer
