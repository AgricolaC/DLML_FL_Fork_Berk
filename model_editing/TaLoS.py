import torch


def compute_fisher_scores(model, dataloader, criterion, device, num_samples=5):
    """
    Compute Fisher Information matrix diagonal elements (sensitivity scores).
    """
    model.eval()
    fisher_scores = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in model.named_parameters() if param.requires_grad}

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            logits = model(inputs)
            dist = torch.distributions.Categorical(logits=logits)
        
        for _ in range(num_samples):
            model.zero_grad()
            samples = dist.sample()
            loss = criterion(logits, samples)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_scores[name] += (param.grad.detach().cpu() ** 2) / len(dataloader)

    return fisher_scores

def calibrate_mask(model, fisher_scores, target_sparsity, rounds=4):
    """
    Gradual mask calibration with layer-wise consideration
    """
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize masks and scores
    masks = {n: torch.ones_like(p, device='cpu') 
            for n, p in model.named_parameters() if p.requires_grad}
    all_scores = torch.cat([v.flatten().cpu() for v in fisher_scores.values()])

    for round in range(rounds):
        # Calculate current sparsity level
        current_sparsity = target_sparsity ** ((round + 1) / rounds)
        
        # Find global threshold
        k = int(current_sparsity * all_scores.numel())
        threshold = torch.kthvalue(all_scores, k).values.item()

        # Update masks
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            score = fisher_scores[name].to(device)
            mask = (score > threshold).float().cpu()
            masks[name] = mask * masks[name]  # Preserve previous masking
            
            # Freeze parameters below threshold
            param.requires_grad = (mask.sum() > 0).item()

        # Remove masked parameters from future consideration
        all_scores = all_scores[all_scores > threshold]

    return {n: m.to(device) for n, m in masks.items()}
