# Intersection Over Union
- Address the problem of "how do we measure how good a bounding box is?"

# How it Works
1. Calculate the area of Intersection with the Prediction and Ground Truth (GT)
2. Divide with area of Union of the Prediction and GT

IOU = Area of Intersection / Area Union

IOU will ranged from 0 to 1

## Rule of Thumb
More than 0.5, decent
0.7, pretty good
0.9, almost perfect

## How do we get the intersection
Box1 = [x1, x2, y1, y2]
Box2 = [x1, x2, y1, y2]
Assumption: The origin is (0, 0)

Box_Intersection_x1 = max(Box1_x1, Box2_x1)
Box_Intersection_y1 = max(Box1_y1, Box2_y1)
Box_Intersection_x2 = min(Box1_x2, Box2_x2)
Box_Intersection_y2 = min(Box1_y2, Box2_y2)