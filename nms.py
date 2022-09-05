import torch
from iou import Intersection_over_union

def nms(bboxes, iou_threshold, threshold, format='corners'):
    
    assert type(bboxes) == list
    
    bboxes = [ box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x:x[1], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [ box for box in bboxes if chosen_box[0] != box[0] or Intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=format)<iou_threshold]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms