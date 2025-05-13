import torch
from torch.utils.checkpoint import checkpoint


def compute_fisher_scores(model, dataloader, criterion, device):
    """
    Compute Fisher Information matrix diagonal elements (sensitivity scores).
    """
    model.eval()
    fisher_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_scores[name] += param.grad ** 2

    # Normalize scores
    for name in fisher_scores:
        fisher_scores[name] /= len(dataloader)

    return fisher_scores


def calibrate_mask(model, fisher_scores, target_sparsity, rounds, dynamic_sparsity=False, layer_wise=False, logger=None, use_checkpointing=False):
    """
    Iteratively calibrate masks based on Fisher scores to reach target sparsity.

     Args:
        model: The model whose parameters are being pruned.
        fisher_scores: Dictionary of Fisher scores for each layer.
        target_sparsity: Global or layer-wise sparsity target (float or dict).
        rounds: Number of calibration rounds.
        dynamic_sparsity: Adjust sparsity dynamically over rounds.
        layer_wise: Enable layer-wise sparsity targets.
        logger: Logger for monitoring progress.
        use_checkpointing: Enable gradient checkpointing for memory efficiency.

    Returns:
        masks: Dictionary of masks for each layer.
    """
    masks = {name: torch.ones_like(param) for name, param in model.named_parameters() if param.requires_grad}
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    
    # layer-wise sparsity initialization
    if layer_wise and isinstance(target_sparsity, dict):
        layer_target_params = {name: int(param.numel() * (1 - target_sparsity.get(name, 0))) for name, param in model.named_parameters() if param.requires_grad}}
    else:
        layer_target_params = None
        target_params = int(total_params * (1 - target_sparsity))      

    for _ in range(rounds):
        if logger:
            logger.info(f"Calibration Round {_ + 1}/{rounds}")
            
        # Flatten scores and masks
        if layer_wise and layer_target_params: 
            thresholds = {}
            for name,param in fisher_scores.items():
                if name in masks: 
                    flattened_scores = params[masks[name]>0].flatten()
                    layer_threshold = torch.topk(flattened_scores, layer_target_params[name], largest=False).values[-1]
                    thresholds[name] = layer_threshold
        else:
            # Apply checkpointing for global scores
            def compute_global_scores():
                return torch.cat([fisher_scores[name][masks[name] > 0].flatten() for name in fisher_scores])
            if use_checkpointing:
                all_scores = checkpoint(compute_global_scores)
            else:
                all_scores = compute_global_scores()
                
            threshold = torch.topk(all_scores, target_params, largest=False).values[-1]

        # Update masks
        for name in masks:
            if layer_wise and thresholds:
                masks[name] = (fisher_scores[name] >= thresholds[name]).float()
            else:
                masks[name] = (fisher_scores[name] >= threshold).float()

         # Dynamic sparsity adjustments
        if dynamic_sparsity and round_idx < rounds - 1:
            if layer_wise and layer_target_params:
                for name in layer_target_params:
                    layer_target_params[name] = int(layer_target_params[name] * (1 - target_sparsity / rounds))
            else:
                target_params = int(target_params * (1 - target_sparsity / rounds))
                
        if logger:
            sparsity_level = {name: 1 - masks[name].mean().item() for name in masks}
            logger.info(f"Current Sparsity Levels: {sparsity_level}")

    return masks
