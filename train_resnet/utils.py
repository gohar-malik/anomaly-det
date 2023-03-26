import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer, SGD
from torch.optim.optimizer import Optimizer, required

class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large batch training of convolutional networks with layer-wise adaptive rate scaling. ICLR'18:
        https://openreview.net/pdf?id=rJ4uaX2aW
    The LARS algorithm can be written as
    .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + (1.0 - \mu) * (g_{t} + \beta * w_{t}), \\
                w_{t+1} & = w_{t} - lr * ||w_{t}|| / ||v_{t+1}|| * v_{t+1},
            \end{aligned}
    where :math:`w`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, dampening = 0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        #if eta < 0.0:
        #    raise ValueError("Invalid eta value:{}".format(eta))

        defaults = dict(lr=lr, momentum = momentum,
                        weight_decay = weight_decay,
                        dampening = dampening)

        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                # gradient
                d_p = p.grad.data
                weight_norm = torch.norm(p.data)

                # update the velocity
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                # l2 regularization
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                buf.mul_(momentum).add_(d_p, alpha = 1.0 - dampening)
                v_norm = torch.norm(buf)

                local_lr = lr * weight_norm / (1e-6 + v_norm)

                # Update the weight
                p.add_(buf, alpha = -local_lr)


        return loss

class AMP(Optimizer):
    """
    Implements adversarial model perturbation.

    Args:
        params (iterable): iterable of trainable parameters
        lr (float): learning rate for outer optimization
        epsilon (float): perturbation norm ball radius
        inner_lr (float, optional): learning rate for inner optimization (default: 1)
        inner_iter (int, optional): iteration number for inner optimization (default: 1)
        base_optimizer (class, optional): basic optimizer class (default: SGD)
        **kwargs: keyword arguments passed to the `__init__` method of `base_optimizer`

    Example:
        >>> optimizer = AMP(model.parameters(), lr=0.1, eps=0.5, momentum=0.9)
        >>> for inputs, targets in dataset:
        >>>     def closure():
        >>>         optimizer.zero_grad()
        >>>         outputs = model(inputs)
        >>>         loss = loss_fn(outputs, targets)
        >>>         loss.backward()
        >>>         return outputs, loss
        >>>     outputs, loss = optimizer.step(closure)
    """

    def __init__(self, params, lr, epsilon, inner_lr=1, inner_iter=1, base_optimizer=SGD, **kwargs):
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if inner_lr < 0.0:
            raise ValueError(f"Invalid inner lr: {inner_lr}")
        if inner_iter < 0:
            raise ValueError(f"Invalid inner iter: {inner_iter}")
        defaults = dict(lr=lr, epsilon=epsilon, inner_lr=inner_lr, inner_iter=inner_iter, **kwargs)
        super(AMP, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, lr=lr, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError('Adversarial model perturbation requires closure, but it was not provided')
        closure = torch.enable_grad()(closure)
        outputs, loss = map(lambda x: x.detach(), closure())
        for i in range(self.defaults['inner_iter']):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if i == 0:
                            self.state[p]['dev'] = torch.zeros_like(p.grad)
                        dev = self.state[p]['dev'] + group['inner_lr'] * p.grad
                        clip_coef = group['epsilon'] / (dev.norm() + 1e-12)
                        dev = clip_coef * dev if clip_coef < 1 else dev
                        p.sub_(self.state[p]['dev']).add_(dev) # update "theta" with "theta+delta"
                        self.state[p]['dev'] = dev
            closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.sub_(self.state[p]['dev']) # restore "theta" from "theta+delta"
        self.base_optimizer.step()
        return outputs, loss
    
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def init_weights(net):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        # print(type(m))
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture
    
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]