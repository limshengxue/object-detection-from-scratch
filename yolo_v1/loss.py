import torch
import torch.nn as nn
from ..iou import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # -1 to keep the number of instances
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # pass in 4 bounding box value
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # pass in 4 bounding box value

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        # 2 ious stack along the 0 dimension
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) # expand the last dim (N, S, S, 1)

        # ================== #
        # FOR BOX COORD      #
        # ================== #
        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] # take the second box
            + (1 - bestbox) * predictions[..., 21:25] # take the first box
        )
        box_targets = exists_box * target[..., 21:25]

        # scale the height and width 
        # abs to prevent negative
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) *torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
            )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim = -2)
        )

        # ================== #
        # FOR OBJECT LOSS     #
        # ================== #
        pred_box = (
            bestbox * predictions[..., 25:26] 
            + (1- bestbox) * predictions[..., 20:21]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ================== #
        # FOR NO OBJECT LOSS     #
        # ================== #
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21] , start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21] , start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * predictions[..., 25:26] , start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21] , start_dim=1),
        )

        # ================== #
        # FOR CLASS LOSS     #
        # ================== #
        # (N, S, S, 20) -> (N * S * S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim = -2),
            torch.flatten(exists_box * target[..., :20], end_dim = -2)
        )

        # ================== #
        # TOTAL LOSS     #
        # ================== #

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
