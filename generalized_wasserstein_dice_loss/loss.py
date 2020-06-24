import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class GeneralizedWassersteinDiceLoss(_Loss):
    """
    Generalized Wasserstein Dice Loss [1] in PyTorch.
    Compared to [1] we used a weighting method similar to the one
    used in the generalized Dice Loss [2].

    References:
    ===========
    [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017.
    [2] "Generalised dice overlap as a deep learning loss function
        for highly unbalanced segmentations",
        Sudre C., et al. MICCAI DLMIA 2017.
    """
    def __init__(self, dist_matrix, reduction='mean'):
        """
        :param dist_matrix: 2d tensor or 2d numpy array; matrix of distances between the classes.
        It must have dimension C x C where C is the number of classes.
        :param reduction: str; reduction mode.
        """
        super(GeneralizedWassersteinDiceLoss, self).__init__(reduction=reduction)
        self.M = dist_matrix
        if isinstance(self.M, np.ndarray):
            self.M = torch.from_numpy(self.M)
        if torch.cuda.is_available():
            self.M = self.M.cuda()
        if torch.max(self.M) != 1:
            print('Normalize the maximum of the distance matrix '
                  'used in the Generalized Wasserstein Dice Loss to 1.')
            self.M = self.M / torch.max(self.M)
        self.num_classes = self.M.size(0)
        self.reduction = reduction

    def forward(self, input, target):
        epsilon = np.spacing(1)  # smallest number available
        # Convert the target segmentation to long if needed
        if not(target.type() in [torch.LongTensor, torch.cuda.LongTensor]):
            target = target.long()
        # Aggregate spatial dimensions
        flat_input = input.view(input.size(0), input.size(1), -1)  # b,c,s
        flat_target = target.view(target.size(0), -1)  # b,s
        # Apply the softmax to the input scores map
        probs = F.softmax(flat_input, dim=1)  # b,c,s
        # Compute the Wasserstein distance map
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)
        # Compute the generalised number of true positives
        alpha = self.compute_weights_generalized_true_positives(flat_target)
        true_pos = self.compute_generalized_true_positive(alpha, flat_target, wass_dist_map)
        denom = self.compute_denominator(alpha, flat_target, wass_dist_map)
        # Compute and return the final loss
        wass_dice = (2. * true_pos + epsilon) / (denom + epsilon)
        wass_dice_loss = 1. - wass_dice
        if self.reduction == 'sum':
            return wass_dice_loss.sum()
        elif self.reduction == 'none':
            return wass_dice_loss
        # default is mean reduction
        else:
            return wass_dice_loss.mean()

    def wasserstein_distance_map(self, flat_proba, flat_target):
        """
        Compute the voxel-wise Wasserstein distance (eq. 6 in [1])
        between the flattened prediction and the flattened labels (ground_truth) with respect
        to the distance matrix on the label space M.
        References:
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
        """
        # Turn the distance matrix to a map of identical matrix
        M_extended = torch.unsqueeze(self.M, dim=0)  # C,C -> 1,C,C
        M_extended = torch.unsqueeze(M_extended, dim=3)  # 1,C,C -> 1,C,C,1
        M_extended = M_extended.expand(
            (flat_proba.size(0), M_extended.size(1), M_extended.size(2), flat_proba.size(2))
        )
        # Expand the feature dimensions of the target
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        flat_target_extended = flat_target_extended.expand(  # b,1,s -> b,C,s
            (flat_target.size(0), M_extended.size(1), flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target_extended, dim=1)  # b,C,s -> b,1,C,s
        # Extract the vector of class distances for the ground-truth label at each voxel
        M_extended = torch.gather(M_extended, dim=1, index=flat_target_extended)  # b,C,C,s -> b,1,C,s
        M_extended = torch.squeeze(M_extended, dim=1)  # b,1,C,s -> b,C,s
        # Compute the wasserstein distance map
        wasserstein_map = M_extended * flat_proba
        # Sum over the classes
        wasserstein_map = torch.sum(wasserstein_map, dim=1)  # b,C,s -> b,s
        return wasserstein_map

    def compute_generalized_true_positive(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s
        # Compute the generalized true positive as in eq. 9
        generalized_true_pos = torch.sum(
            alpha_extended * (1. - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    def compute_denominator(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s
        # Compute the generalized true positive as in eq. 9
        generalized_true_pos = torch.sum(
            alpha_extended * (2. - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    def compute_weights_generalized_true_positives(self, flat_target):
        """
        Compute the weights \alpha_l of eq. 9 in [1] but using the weighting
        method proposed in the generalized Dice Loss [2].
        References:
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
        [2] "Generalised dice overlap as a deep learning loss function
        for highly unbalanced segmentations." Sudre C., et al.
        MICCAI DLMIA 2017.
        """
        # Convert target to one-hot class encoding
        one_hot = F.one_hot(  # shape: b,c,s
            flat_target, num_classes=self.num_classes).permute(0, 2, 1).float()
        volumes = torch.sum(one_hot, dim=2)  # b,c
        alpha = 1. / (volumes + 1.)
        return alpha
