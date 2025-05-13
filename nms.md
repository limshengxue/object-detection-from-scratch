# Non Maximum Suppression
- To clean up multiple overlapping bounding box regarding a single object
- NMS will perform separately for instance of different class

# How it works
- Discard some bounding box with probability lower than a certain threshold
- Take out the highest scoring box (the bbox that has the highest probability containing the object) 
- Compute the IOU between the highest scoring box and others overlapping boxes
- If the IOU is higher than certain threshold, remove the overlapping box