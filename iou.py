import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    # boxes_preds shape is (N, 4) where N is the number of bboxes
    # boxes_labels shape is (N, 4)
    """
    Calculates the Intersection over Union (IoU) for two sets of bounding boxes.

    Args:
        boxes_preds (torch.Tensor): Predicted bounding boxes, shape (N, 4), where N is the number of boxes.
        boxes_labels (torch.Tensor): Ground truth bounding boxes, shape (N, 4).
        box_format (str): Format of the bounding boxes, either "midpoint" or "corners".

    Returns:
        torch.Tensor: IoU for each pair of bounding boxes, with values ranging from 0 to 1.
                      Higher values indicate better overlap.

    The function supports two formats for the bounding box representation:
    - "midpoint": [x_center, y_center, width, height]
    - "corners":  [x1, y1, x2, y2]
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] # (N, 1)

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    intersection_x1 = torch.max(box1_x1, box2_x1)
    intersection_y1 = torch.max(box1_y1, box2_y1)
    intersection_x2 = torch.min(box1_x2, box2_x2)
    intersection_y2 = torch.min(box1_y2, box2_y2)

    # clamp to handle the case where there is no intersection
    area_of_intersection = (intersection_x2 - intersection_x1).clamp(0) * (intersection_y2 - intersection_y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return area_of_intersection / (box1_area + box2_area - area_of_intersection + 1e-6)
   

    
    



