# Generalized Wasserstein Dice Loss
The [Generalized Wasserstein Dice Loss][brainles17] (GWDL) is a loss function to train deep neural networks
for applications in medical image multi-class segmentation.

The GWDL is a generalization of the Dice loss and the [Generalized Dice loss][gdl17] 
that can tackle hierarchical classes and can take advantage of known relationships between classes.

## Installation
```bash
pip install git+https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss.git
```

## Example
```python
import torch
import numpy as np
from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss

# Example with 3 classes (including the background: label 0).
# The distance between the background (class 0) and the other classes is the maximum, equal to 1.
# The distance between class 1 and class 2 is 0.5.
dist_mat = np.array([
    [0., 1., 1.],
    [1., 0., 0.5],
    [1., 0.5, 0.]
])
wass_loss = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)
# 1D prediction; shape: batch size, n class, n elements
pred = torch.tensor([[[1, 0], [0, 1], [0, 0]]], dtype=torch.float32).cuda()
# !D ground truth; shape: batch size, n elements 
grnd = torch.tensor([[0, 2]], dtype=torch.int64).cuda()
wass_loss(pred, grnd)
```

## How to cite
If you use the Generalized Wasserstein Dice Loss in your work,
please cite
* L. Fidon, W. Li, L. C. Garcia-Peraza-Herrera, J. Ekanayake, N. Kitchen, S. Ourselin, T. Vercauteren.
[Generalised Wasserstein Dice Score for Imbalanced Multi-class Segmentation using Holistic Convolutional Networks.][brainles17]
International MICCAI Brainlesion Workshop. Springer, Cham, 2017.

BibTeX:
```
@inproceedings{fidon2017generalised,
  title={Generalised {W}asserstein dice score for imbalanced multi-class segmentation using holistic convolutional networks},
  author={Fidon, Lucas and Li, Wenqi and Garcia-Peraza-Herrera, Luis C and Ekanayake, Jinendra and Kitchen, Neil and Ourselin, S{\'e}bastien and Vercauteren, Tom},
  booktitle={International MICCAI Brainlesion Workshop},
  pages={64--76},
  year={2017},
  organization={Springer}
}
```

## Applications of the Generalized Wasserstein Dice loss
For more examples of applications of the generalized Wasserstein Dice loss and how to define the distance matrix, you can look at:
- more on brain tumor segmentation from MRI
[Fidon, Lucas, Sebastien Ourselin, and Tom Vercauteren. "Generalized Wasserstein Dice Score, Distributionally Robust Deep Learning, and Ranger for brain tumor segmentation: BraTS 2020 challenge." arXiv preprint arXiv:2011.01614 (2020).](https://arxiv.org/abs/2011.01614)
- segmentation of lung lesions due to COVID-19 from CT (see the Appendix)
[Tilborghs, Sofie, et al. "Comparative study of deep learning methods for the automatic segmentation of lung, lesion and lesion type in CT scans of COVID-19 patients." arXiv preprint arXiv:2007.15546 (2020).](https://arxiv.org/abs/2007.15546)
- segmentation of subcutaneous adipose tissue, visceral adipose tissue, muscular body mass, bone, and visceral organs from low-dose CT images (see the supplementary material)
[Blanc-Durand, Paul, et al. "Prognostic value of anthropometric measures extracted from whole-body CT using deep learning in patients with non-small-cell lung cancer." European Radiology (2020): 1-10.](https://link.springer.com/article/10.1007%2Fs00330-019-06630-w)

If you find more papers using the generalized Wasserstein Dice loss please let me know :)

[brainles17]: https://arxiv.org/abs/1707.00478
[gdl17]: https://arxiv.org/abs/1707.03237