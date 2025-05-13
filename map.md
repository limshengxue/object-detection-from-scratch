# Mean Average Precision
- The most common metric used in DL to evaluate object detection model

# How it Works
1. Get all bbox predictions (decide if it is a TP or FP based on IOU thresholding with the GT)
2. Sort by descending confidence score
3. Calculate Precision and Recall as we go through all bbox output
4. Plot Precision-Recall graph
5. Take the AUC of the PR graph
6. Do it for every class and take the mean
7. Repeat for multiple IOU and take the average

mAP@0.5:0.05:0.95 indicates the computation starts with IOU 0.5 with step size 0.05 up until 0.95