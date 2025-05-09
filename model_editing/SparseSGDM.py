import torch
from torch.optim import SGD


class SparseSGDM(SGD):
    def __init__(self, params, lr, momentum=0, weight_decay=0, masks=None):
        super().__init__(params, lr, momentum=momentum, weight_decay=weight_decay)
        self.masks = masks or {}
        self._init_momentum_buffers()

    def _init_momentum_buffers(self):
        """Initialize masked momentum buffers"""
        for group in self.param_groups:
            for p in group['params']:
                if p in self.masks and 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p not in self.masks:
                    continue
                
                # Apply mask to gradient
                p.grad *= self.masks[p].to(p.device)
                
                # Apply mask to momentum buffer
                if 'momentum_buffer' in self.state[p]:
                    self.state[p]['momentum_buffer'] *= self.masks[p].to(p.device)
        
        return super().step(closure)
