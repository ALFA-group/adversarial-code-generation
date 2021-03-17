from __future__ import print_function
import math
import torch
import torch.nn as nn
import numpy as np

class Loss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def backward(self, retain_graph=False):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward(retain_graph=retain_graph)


class NLLLoss(Loss):
    """ Batch averaged negative log-likelihood loss.
    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Avg NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        reduction = 'mean' if self.size_average else 'sum'

        super(NLLLoss, self).__init__(self._NAME,nn.NLLLoss(weight=weight, reduction=reduction))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data.item()
        # if self.size_average:
        #     # average loss per batch
        #     loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target, weight=1.0):
        self.acc_loss += weight*self.criterion(outputs, target)
        self.norm_term += 1

class Perplexity(NLLLoss):
    """ Language model perplexity loss.

    Perplexity is the token averaged likelihood.  When the averaging options are the
    same, it is the exponential of negative log-likelihood.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    _NAME = "Perplexity"
    _MAX_EXP = 100

    def __init__(self, weight=None, mask=None, reduction='none'):
        super(Perplexity, self).__init__(weight=weight, mask=mask, size_average=True)

    def eval_batch(self, outputs, target, weight=1.0):
        # shape outputs: batch_sz x decoded output dims
        # shape target: batch_sz x 1
        ww = self.criterion(outputs, target)
        self.acc_loss += weight*ww
        if self.mask is None:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        nll = super(Perplexity, self).get_loss()
        nll /= self.norm_term.item()
        if nll > Perplexity._MAX_EXP:
            print("WARNING: Loss exceeded maximum value, capping to e^100")
            return math.exp(Perplexity._MAX_EXP)
        return math.exp(nll)

class AttackLoss():
    _NAME = 'Attack loss'
    def __init__(self, device, tau=0):
        self.TAU = torch.empty((1,), device=device, dtype=torch.half).fill_(-1*tau)
    
    def get_loss(self, predicted_logits, target):
        # Accumulate loss over all tokens in the output
        loss_batch = None
        for step, step_output in enumerate(predicted_logits):
            # Expected size of pred: batch_sz x output vocab
            # Expected size of actual: batch_sz x (# of output tokens + 1)
            pred, actual = step_output.contiguous().view(target.size(0), -1), target[:, step + 1]
            pred_clone = pred.clone()
            pred_actual = None
            for cnt in range(pred_clone.shape[0]):
                pred_clone[cnt, actual[cnt]] = -1000
                if pred_actual is None:
                    pred_actual = pred[cnt, actual[cnt]].unsqueeze(dim=0)
                else:
                    pred_actual = torch.cat((pred_actual, pred[cnt, actual[cnt]].unsqueeze(dim=0)), 0)
            l = torch.max(pred_actual - torch.max(pred_clone, dim=1).values, self.TAU)
            if loss_batch is None:
                loss_batch = l.item()
            else:
                loss_batch += l.item()

        loss_sum = torch.sum(loss_batch)
        #print('Loss {}'.format(loss_sum.shape))
        #print('Loss-batch {}'.format(loss_batch.shape))
        return loss_sum, loss_batch
