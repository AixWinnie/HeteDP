import torch
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F

from typing import Optional
from gaussian_accountant import create_accountant

def feat_noise(features, delta, sens_size, epsilon):
    sigma = torch.sqrt(2 * torch.log(1.25 / delta)) * sens_size / epsilon
    noise = sigma * torch.normal(mean=torch.zeros_like(features).data,std=torch.tensor(1.0).cuda())
    return features + 0.01 * noise

def clip_and_accumulate(args, model):
    batch_size = args.batch_proc_size
    cum_grads = model.cum_grads
    C = args.grad_norm_max
    """
    Performs gradient clipping.
    Stores clipped and aggregated gradients.
    """
    g_norm = Variable(torch.zeros(batch_size), requires_grad=False)
    counter2 = 0
    g_norm = {}
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        counter2 += 1
        if p.grad is not None:
            g_norm[str(counter2)] = p.grad.norm(2, dim=-1)      

    for p, key in zip(filter(lambda p: p.requires_grad, model.parameters()), cum_grads.keys()):
        if p.grad is not None and key in g_norm.keys():
            cum_grads[key] = F.normalize(torch.sum((p.grad / torch.clamp(g_norm[key].contiguous().view(-1, 1) / C, min=1)), dim=0),dim=0)

def add_noise(args, model):
    cum_grads = model.cum_grads
    C = args.grad_norm_max
    """
    Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
    """
    for p, key in zip(filter(lambda p: p.requires_grad, model.parameters()), cum_grads.keys()):
        if p.grad is not None:
            proc_size = p.grad.size(0)
            '''
            add noise to summed clipped pars
            compute sigma
            '''
            noise_multiplier = get_noise_multiplier(
                            target_epsilon=args.eps_requirement,
                            target_delta=args.delta,
                            sample_rate=args.sample_rate,
                            epochs=args.epochs,
                            accountant="gdp",
                        )
            noise = _generate_noise(
                std=noise_multiplier * C,
                reference=cum_grads[key]
            )
            noise = F.normalize(noise,dim=0)
            if len(p.data.shape) > 1:
                cum_grads[key] = cum_grads[key].expand(proc_size, -1)
            if p.grad.is_cuda:
                p.grad = ((cum_grads[key] + noise).view_as(p.grad)).cuda()/proc_size
            else:
                p.grad = ((cum_grads[key] + noise).view_as(p.grad))/proc_size

def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
) -> torch.Tensor:
    
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    
    return torch.normal(
        mean=0,
        std=std,
        size=reference.shape,
        device=reference.device,
        generator=generator,
    )


def create_cum_grads(model):
    cum_grads = OrderedDict()
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]), requires_grad=False)
    return cum_grads


MAX_SIGMA = 1e6
def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
    **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    if (steps is None) == (epochs is None):
        raise ValueError(
            "get_noise_multiplier takes as input EITHER a number of steps or a number of epochs"
        )
    if steps is None:
        steps = int(epochs / sample_rate)

    eps_high = float("inf")
    accountant = create_accountant(mechanism=accountant)

    sigma_low, sigma_high = 0, 10
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        accountant.history = [(sigma_high, sample_rate, steps)]
        eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        accountant.history = [(sigma, sample_rate, steps)]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high
