import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from transformers import get_scheduler
 
import torch
import torch.nn as nn
import torch.nn.functional as F

from scer.models import networks
from scer.learning import joint_dro
from scer.learning.optimizers import get_optimizers
from scer.utils.misc import mixup_data


ALGORITHMS = [
    'ERM',
    # subgroup methods
    'GroupDRO',
    
    'SCER']

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a subgroup robustness algorithm.
    Subclasses should implement the following:
    - _init_model()
    - _compute_loss()
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_examples = num_examples

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step):
        """Perform one update step."""
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a

        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)

        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ERM, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.featurizer = networks.Featurizer(data_type, input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_model()

    def _init_model(self):
        self.clip_grad = (self.data_type == "text" and self.hparams["optimizer"] == "adamw")

        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.data_type == "text":
            self.network.zero_grad()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)



class GroupDRO(ERM):
    """
    Group DRO minimizes the error at the worst group [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(GroupDRO, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.register_buffer(
            "q", torch.ones(self.num_classes * self.num_attributes).cuda())

    def _compute_loss(self, i, x, y, a, step):
        losses = self.loss(self.predict(x), y)

        for idx_g, idx_samples in self.return_groups(y, a):
            self.q[idx_g] *= (self.hparams["groupdro_eta"] * losses[idx_samples].mean()).exp().item()

        self.q /= self.q.sum()

        loss_value = 0
        for idx_g, idx_samples in self.return_groups(y, a):
            loss_value += self.q[idx_g] * losses[idx_samples].mean()

        return loss_value



class SCER(ERM):
    """
        SCER with product-based constraints:
      1) ||Delta_0||_Sigma * (Delta_0)^T B -> 0
      2) ||Delta||_Sigma * (Delta)^T B -> large (>0)
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes,
                 num_examples, hparams, grp_sizes=None):
        super(EDRO3, self).__init__(
            data_type, input_shape, num_classes, num_attributes,
            num_examples, hparams, grp_sizes
        )
        self.register_buffer("q", torch.ones(self.num_classes * self.num_attributes).cuda())
        self.sigma_matrix = None
        self.embedding_loss_before_step = torch.tensor(0.0, device='cuda')

    def compute_sigma_matrix(self, batch_embeddings):
        mean_embedding = torch.mean(batch_embeddings, dim=0, keepdim=True)
        deviations = batch_embeddings - mean_embedding
        covariance_matrix = torch.matmul(deviations.T, deviations) / (batch_embeddings.size(0) - 1)
        num_features = batch_embeddings.shape[1]
        # Simple regularization
        sigma_matrix = (covariance_matrix / (1e-6 + torch.norm(covariance_matrix)) * 0.1 +
                        torch.eye(num_features).cuda())
        return sigma_matrix
    @staticmethod
    def compute_a0(group_counts):
        """
        group_counts : 
        {0: num_samples_in_G0,
        1: num_samples_in_G1,
        2: num_samples_in_G2,
        3: num_samples_in_G3}
        G0=(y=0,a=0), G1=(y=0,a=1), G2=(y=1,a=0), G3=(y=1,a=1).
        """
        total = sum(group_counts.values())
        
        # y=1 -> (G2, G3), y=0 -> (G0, G1)
        label1_count = group_counts.get(2, 0) + group_counts.get(3, 0)
        label0_count = group_counts.get(0, 0) + group_counts.get(1, 0)

        # p(G=1) = P(y=1) = (G2 + G3) / total
        pG1 = label1_count / (total + 1e-12)
        # p(G=0) = P(y=0) = (G0 + G1) / total
        pG0 = label0_count / (total + 1e-12)

        # p(R=1|G=1) = P(a=1 | y=1) = G3 / (G2 + G3)
        pR1_givenG1 = group_counts.get(3, 0) / (label1_count )
        # p(R=0|G=0) = P(a=0 | y=0) = G0 / (G0 + G1)
        pR0_givenG0 = group_counts.get(0, 0) / (label0_count )

        # a0 = 1/2 [ p(R=1|G=1)*p(G=1) + p(R=0|G=0)*p(G=0) ]
        a0 = 0.5 * (pR1_givenG1 * pG1 + pR0_givenG0 * pG0)
        return a0

    def _compute_loss(self, i, x, y, a, step):
        # 1. Get embeddings and update sigma_matrix
        batch_embeddings = self.return_feats(x)
        self.sigma_matrix = self.compute_sigma_matrix(batch_embeddings)
        Sigma_inv = torch.linalg.inv(self.sigma_matrix) 

        # 2. Standard classification / DRO steps
        predictions = self.predict(x)
        losses = self.loss(predictions, y)
        for idx_g, idx_samples in self.return_groups(y, a):
            self.q[idx_g] *= (self.hparams["scer_eta"] * losses[idx_samples].mean()).exp().item()
        self.q /= self.q.sum()
        dro_loss = sum(
            self.q[idx_g] * losses[idx_samples].mean()
            for idx_g, idx_samples in self.return_groups(y, a)
        )

        # 3. Group embeddings => define Delta_0 and Delta
        group_embeddings_dict = {}
        group_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for idx_g, idx_samples in self.return_groups(y, a):
            if len(idx_samples) == 0:
                continue
            group_x = x[idx_samples]
            emb = self.return_feats(group_x)
            group_embeddings_dict[idx_g.item()] = emb.mean(dim=0)
            group_counts[idx_g.item()] += idx_samples.sum().item()


        if all(g in group_embeddings_dict for g in [0, 1, 2, 3]):
            G0 = group_embeddings_dict[0]
            G1 = group_embeddings_dict[1]
            G2 = group_embeddings_dict[2]
            G3 = group_embeddings_dict[3]

            delta0_vec = 0.5 * (abs(G1 - G0) + abs(G3 - G2))
            delta_vec = 0.5 * (abs(G2 - G0) + abs(G3 - G1))
            delta0_vec = delta0_vec / (1e-6 + delta0_vec.norm(p=2))
            delta_vec = delta_vec / (1e-6 + delta_vec.norm(p=2))

            delta0_norm = torch.sqrt(delta0_vec @ self.sigma_matrix @ delta0_vec)
            delta_norm = torch.sqrt(delta_vec @ self.sigma_matrix @ delta_vec)
            
            a0 = self.compute_a0(group_counts)

 
            numerator = a0 * (delta_vec @ Sigma_inv @ delta0_vec)
            denominator = 1.0 + a0 * (delta0_vec @ Sigma_inv @ delta0_vec)
            Beta = Sigma_inv @ delta_vec - Sigma_inv @ delta0_vec * (numerator / denominator)
            Beta_norm = torch.sqrt(Beta@ self.sigma_matrix @ Beta) 


            dot_delta0_B = Beta @ delta0_vec.T
            dot_delta0_B = dot_delta0_B /  (delta0_norm *Beta_norm)
            dot_delta_B = Beta @ delta_vec.T
            dot_delta_B = dot_delta_B / (delta_norm *Beta_norm) 

            spurious_product = delta0_norm * dot_delta0_B
            real_product = delta_norm * dot_delta_B

            spurious_loss = abs(spurious_product)

            embedding_loss = (self.hparams["lambda_spurious"] * spurious_loss -
            self.hparams["lambda_core"] * real_product)
            self.embedding_loss_before_step = embedding_loss.clone().detach().requires_grad_(True)
        else:
            embedding_loss = self.embedding_loss_before_step


        self.embedding_loss_before_step = embedding_loss.clone().detach().requires_grad_(True)


        total_loss = dro_loss + embedding_loss
        return total_loss


