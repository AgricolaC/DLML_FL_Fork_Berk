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

def calibrate_mask(model, fisher_scores, target_sparsity, rounds):
    """
    Iteratively calibrate masks based on Fisher scores to reach target sparsity.
    """
    masks = {name: torch.ones_like(param) for name, param in model.named_parameters() if param.requires_grad}
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    target_params = int(total_params * (1 - target_sparsity))       # params to be kept

    for _ in range(rounds):
        # Flatten scores and masks
        all_scores = torch.cat([fisher_scores[name][masks[name] > 0].flatten() for name in fisher_scores])
        threshold = torch.topk(all_scores, target_params, largest=False).values[-1]

        # Update masks
        for name in masks:
            masks[name] = (fisher_scores[name] <= threshold).float()

        # Update target params for next round
        target_params = int(target_params * (1 - target_sparsity / rounds))

    return masks
