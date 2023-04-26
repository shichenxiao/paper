import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class PGD(nn.Module):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - input_tensor: :math:`(N, L, d)`
        - labels: :
        - output: :

    Examples::

    """
    def __init__(self,
                 eps=0.08,
                 alpha=0.05,
                 steps=3,
                 ):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def forward(self, input_embedding, input_tensor_b, model=None):
        r"""

        """
        model.eval()

        input_embedding = input_embedding.clone().detach()
        input_tensor_b = input_tensor_b.clone().detach()
        adv_inputs = input_embedding.clone().detach()

        loss = compute_cl_loss
        for _ in range(self.steps):
            adv_inputs.requires_grad = True
            input_tensor_a_ = model.get_input_representations(inputs_embeds=adv_inputs)

            # Calculate loss
            cost = - loss(input_tensor_a_, input_tensor_b)

            # Update adversarial samples
            grad = torch.autograd.grad(cost, adv_inputs,
                                       retain_graph=False, create_graph=False)[0]

            adv_inputs = adv_inputs.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_inputs - input_embedding, min=-self.eps, max=self.eps)
            adv_inputs = torch.clamp(input_embedding + delta, min=0, max=1).detach()

        return adv_inputs
