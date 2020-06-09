import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss


class TestGeneralizedWassersteinDiceLoss(unittest.TestCase):
    def test_bin_seg_2d(self):
        M = np.array(
            [[0,1],
             [1,0]]
        )
        # define 2d examples
        target = torch.tensor(
            [[0,0,0,0],
             [0,1,1,0],
             [0,1,1,0],
             [0,0,0,0]]
        )
        if torch.cuda.is_available():
            target = target.cuda()

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(
            target, num_classes=2).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(
            1 - target, num_classes=2).permute(0, 3, 1, 2).float()

        # initialize the loss
        loss = GeneralizedWassersteinDiceLoss(dist_matrix=M)

        # the loss for pred_very_good should be close to 0
        loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(loss_good, 0., places=3)

        # same test, but with target with a class dimension
        target_4dim = target.unsqueeze(1)  # shape (1, 1, H, W)
        loss_good = float(loss.forward(pred_very_good, target_4dim).cpu())
        self.assertAlmostEqual(loss_good, 0., places=3)

        # the loss for pred_very_poor should be close to 1
        loss_poor = float(loss.forward(pred_very_poor, target).cpu())
        self.assertAlmostEqual(loss_poor, 1., places=3)

    def test_empty_class_2d(self):
        num_classes = 2
        M = np.array(
            [[0,1],
             [1,0]]
        )
        # define 2d examples
        target = torch.tensor(
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        if torch.cuda.is_available():
            target = target.cuda()

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(
            target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(
            1 - target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # initialize the loss
        loss = GeneralizedWassersteinDiceLoss(dist_matrix=M)

        # loss for pred_very_good should be close to 0
        loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(loss_good, 0., places=3)

        # loss for pred_very_poor should be close to 1
        loss_poor = float(loss.forward(pred_very_poor, target).cpu())
        self.assertAlmostEqual(loss_poor, 1., places=3)

    def test_bin_seg_3d(self):
        M = np.array(
            [[0,1],
             [1,0]]
        )
        # define 3d examples
        target = torch.tensor(
            [
            # raw 0
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            # raw 1
             [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]],
            # raw 2
             [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]]
             ]
        )
        if torch.cuda.is_available():
            target = target.cuda()

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W, D)
        pred_very_good = 1000 * F.one_hot(
            target, num_classes=2).permute(0, 4, 1, 2, 3).float()
        pred_very_poor = 1000 * F.one_hot(
            1 - target, num_classes=2).permute(0, 4, 1, 2, 3).float()

        # initialize the loss
        loss = GeneralizedWassersteinDiceLoss(dist_matrix=M)

        # mean dice loss for pred_very_good should be close to 0
        loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(loss_good, 0., places=3)

        # mean dice loss for pred_very_poor should be close to 1
        loss_poor = float(loss.forward(pred_very_poor, target).cpu())
        self.assertAlmostEqual(loss_poor, 1., places=3)

    def test_convergence(self):
        """
        The goal of this test is to assess if the gradient of the loss function
        is correct by testing if we can train a one layer neural network
        to segment one image.
        We verify that the loss is decreasing in almost all SGD steps.
        """
        M = np.array(
            [[0,1],
             [1,0]]
        )
        learning_rate = 0.01
        max_iter = 50

        # define a simple 3d example
        target_seg = torch.tensor(
            [
            # raw 0
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            # raw 1
             [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]],
            # raw 2
             [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]]
             ]
        )
        if torch.cuda.is_available():
            target_seg = target_seg.cuda()
        target_seg = torch.unsqueeze(target_seg, dim=0)
        image = 12 * target_seg + 27
        image = image.float()
        num_classes = 2
        num_voxels = 3 * 4 * 4
        # define a one layer model
        class OnelayerNet(nn.Module):
            def __init__(self):
                super(OnelayerNet, self).__init__()
                self.layer = nn.Linear(num_voxels, num_voxels * num_classes)
            def forward(self, x):
                x = x.view(-1, num_voxels)
                x = self.layer(x)
                x = x.view(-1, num_classes, 3, 4, 4)
                return x

        # initialise the network
        net = OnelayerNet()
        if torch.cuda.is_available():
            net = net.cuda()

        # initialize the loss
        loss = GeneralizedWassersteinDiceLoss(dist_matrix=M)

        # initialize an SGD
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

        # initial difference between pred and target
        pred_start = torch.argmax(net(image), dim=1)
        diff_start = torch.norm(pred_start.float() - target_seg.float())

        loss_history = []
        # train the network
        for _ in range(max_iter):
            # set the gradient to zero
            optimizer.zero_grad()

            # forward pass
            output = net(image)
            loss_val = loss(output, target_seg)

            # backward pass
            loss_val.backward()
            optimizer.step()

            # stats
            loss_history.append(loss_val.item())

        # difference between pred and target after training
        pred_end = torch.argmax(net(image), dim=1)
        diff_end = torch.norm(pred_end.float() - target_seg.float())

        # count the number of SGD steps in which the loss decreases
        num_decreasing_steps = 0
        for i in range(len(loss_history) - 1):
            if loss_history[i] > loss_history[i+1]:
                num_decreasing_steps += 1
        decreasing_steps_ratio = float(num_decreasing_steps) / (len(loss_history) - 1)

        # verify that the loss is decreasing for sufficiently many SGD steps
        self.assertTrue(decreasing_steps_ratio > 0.9)

        # check that the predicted segmentation has improved
        self.assertGreater(diff_start, diff_end)


if __name__ == '__main__':
    unittest.main()
