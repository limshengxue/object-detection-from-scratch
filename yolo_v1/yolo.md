# YOLO Algorithm
- Split an image into S x S grid
- Each cell output a prediction 
- Target: find the cell contains the object's midpoint
- Each cell left top is the origin (0,0) and right-bottom (1,1)
- Each output and label is relative to the cell (x and y are values between 0 and 1, while w and h can be greater than 1 if the object is larger than the cell)
- The bbox contains (x,y,w,h) which the x,y is the midpoint of the object, and w and h is the width and height of the object
- Label of a cell contains (c1, c2, ..., c20, pc, x, y, w, h) pc - probability of having an object, c = probability for each class
- Prediction looks similar to label but will output 2 bounding box with probability score (1 specialise in wide vs tall)
- Dimension of Label = (S, S, 25) ; Dimension of Prediction = (S, S, 30)
- Limitation: 1 cell can only detect 1 object (finer grid is required when there is more objects)

# Dataset
- Trained on PASCAL VOC Dataset / MS COCO (new)