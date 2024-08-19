import torch
import math


class Lamb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = True
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                m_t = exp_avg / bias_correction1
                v_t = exp_avg_sq / bias_correction2
                mean_v_t = torch.mean(v_t).sqrt()
                torch.sqrt_(v_t)

                # use arctan to do numerically stable division
                # update = m_t / (v_t + group['eps'])
                update = 4 / math.pi * 8 * torch.arctan2(m_t, v_t * 8)

                ratio = 1.0
                if group['weight_decay'] > 0:
                    wd_update = 4 / math.pi * 8 * torch.arctan2(p.data * group['weight_decay'], mean_v_t * 8)
                    update.add_(wd_update)

                    g_norm = torch.norm(update.flatten())
                    w_norm = torch.norm(p.data.flatten())

                    if w_norm > 0.0 and g_norm > 0.0:
                        ratio = w_norm / g_norm

                p.data.add_(update, alpha=-group['lr'] * ratio)

        return loss
