import torch


def compute_fisher_scores(model, dataloader, criterion, device, num_samples=5):
    """
    Compute Fisher Information matrix diagonal elements (sensitivity scores).
    Assumes criterion(logits, target) computes the negative log-likelihood.
    For example, nn.CrossEntropyLoss for classification.
    """
    model.eval() 
    
    fisher_scores = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in model.named_parameters() if param.requires_grad}
    
    # Ensure model parameters require gradients for the Fisher computation
    original_param_grad_states = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_param_grad_states[name] = param.requires_grad
        param.requires_grad_(True) # Temporarily enable grad for all params involved

    num_batches = len(dataloader)
    if num_batches == 0:
        # Handle empty dataloader case
        for name, param in model.named_parameters(): # Restore original grad states
            if name in original_param_grad_states:
                param.requires_grad_(original_param_grad_states[name])
            else: # If it didn't require grad initially and we forced it
                param.requires_grad_(False)
        return fisher_scores

    for inputs, labels in dataloader: 
        inputs = inputs.to(device)
        # Forward pass to get logits - this needs to track gradients
        logits = model(inputs) 
        dist = torch.distributions.Categorical(logits=logits)
        
        for _ in range(num_samples):
            model.zero_grad()
            samples = dist.sample() 
            loss = criterion(logits, samples)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Accumulate squared gradients
                    # Averaging factor will be applied at the end or per addition
                    fisher_scores[name] += (param.grad.detach().cpu() ** 2)
    # we average over the dataset (num_batches) and over the samples from posterior (num_samples)
    for name in fisher_scores:
        fisher_scores[name] /= (num_batches * num_samples)
    # Restore original requires_grad states for parameters
    for name, param in model.named_parameters():
        if name in original_param_grad_states:
            param.requires_grad_(original_param_grad_states[name])
        else: # If it didn't require grad initially and we forced it
             param.requires_grad_(False)
    return fisher_scores
    
def calibrate_mask(model, fisher_scores, target_density_to_keep, rounds=4): # Renamed for clarity
    """
    Gradual mask calibration with layer-wise consideration.
    target_density_to_keep: The final fraction of weights to keep (e.g., 0.1 for 10% dense).
    """
    device = next(model.parameters()).device # Get device from model
    
    # Initialize masks as all ones (keep all initially)
    masks = {name: torch.ones_like(param, device='cpu') 
             for name, param in model.named_parameters() if param.requires_grad}
    
    # Flatten all Fisher scores for global thresholding
    # Ensure only scores for parameters that require grad are included
    # And handle cases where a fisher_score might be empty if a param was unexpectedly filtered
    flat_scores_list = []
    for name, p_scores in fisher_scores.items():
        # Check if the corresponding parameter actually exists and requires grad,
        # to match the parameters for which masks are being created.
        param_exists_and_requires_grad = False
        for model_name, model_param in model.named_parameters():
            if model_name == name and model_param.requires_grad:
                param_exists_and_requires_grad = True
                break
        if param_exists_and_requires_grad:
            flat_scores_list.append(p_scores.flatten().cpu())

    if not flat_scores_list: # No scores to process (e.g., no params require grad)
        return {name: m.to(device) for name, m in masks.items()} # Return initial all-one masks

    all_scores_for_thresholding = torch.cat(flat_scores_list)
    initial_total_params = all_scores_for_thresholding.numel()

    if initial_total_params == 0: # Avoid division by zero or empty tensor issues
         return {name: m.to(device) for name, m in masks.items()}


    for r in range(rounds):
        # Calculate current target density for this round
        # This schedule goes from a higher density to the target_density_to_keep
        # e.g. if target_density_to_keep = 0.1, rounds = 4
        # r=0: current_round_density_target = 0.1^(1/4) ~ 0.56 (keep 56%)
        # r=1: current_round_density_target = 0.1^(2/4) ~ 0.316 (keep 31.6%)
        # r=3: current_round_density_target = 0.1^(4/4) = 0.1 (keep 10%)
        current_round_density_target = target_density_to_keep ** ((r + 1.0) / rounds)
        
        # Number of parameters to keep globally based on current_round_density_target
        # This applies to the *original* total number of parameters, not the shrinking pool
        # Alternatively, if applying to shrinking pool: k = int(current_round_density_target * all_scores_for_thresholding.numel())
        # Let's stick to global percentile for now, which is more stable.
        # This means the threshold is based on the distribution of *all initial scores*.
        
        num_params_to_keep_overall = int(current_round_density_target * initial_total_params)

        if num_params_to_keep_overall == 0 : # Prune everything
            threshold = float('inf') # All scores will be less than this
        elif num_params_to_keep_overall >= initial_total_params: # Keep everything
            threshold = -float('inf') # All scores will be greater
        else:
            # We want to keep the top `num_params_to_keep_overall` scores.
            # So, the threshold is the (total_params - num_params_to_keep_overall)-th smallest score.
            # (or kth largest where k = num_params_to_keep_overall)
            # kthvalue finds the k-th SMALLEST. So if we want to keep K largest, we need N-K th smallest.
            # If k_val is 1-indexed, use initial_total_params - num_params_to_keep_overall for the threshold.
            # If k_val is 0-indexed, use initial_total_params - num_params_to_keep_overall -1.
            # torch.kthvalue is 1-indexed for k.
            k_for_threshold = max(1, initial_total_params - num_params_to_keep_overall) # Ensure k is at least 1
            threshold = torch.kthvalue(all_scores_for_thresholding, k_for_threshold).values.item()

        new_all_scores_for_thresholding_list = []
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in fisher_scores:
                continue
            
            score_tensor = fisher_scores[name].cpu() # Use original scores, not the shrinking `all_scores_for_thresholding`
            
            # Mask elements with scores <= threshold (keep those > threshold)
            # For tie-breaking, >= might be preferred if threshold is an actual score value
            current_param_mask = (score_tensor > threshold).float()
            
            masks[name] = masks[name] * current_param_mask # Iteratively apply mask

            # Optional: Update param.requires_grad (consider if really needed)
            # if masks[name].sum() == 0:
            #     param.requires_grad_(False) # Mark as not requiring grad if fully masked

            # For next round's potential threshold recalculation based on *remaining* scores:
            # (This part is if you DONT use initial_total_params for thresholding each round)
            # relevant_scores = score_tensor[masks[name].bool()] # only scores of weights currently kept
            # new_all_scores_for_thresholding_list.append(relevant_scores.flatten())

        # If thresholding based on remaining parameters:
        # if new_all_scores_for_thresholding_list:
        #    all_scores_for_thresholding = torch.cat(new_all_scores_for_thresholding_list)
        # else: # all params pruned
        #    break 

    return {name: m.to(device) for name, m in masks.items()}
