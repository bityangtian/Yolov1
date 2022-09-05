import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from data import VOCDataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import Yololoss
import time

seed =123
torch.manual_seed(seed)
lr = 2e-5
device ="cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
weight_decay = 0
EPOCHS = 100
num_workers = 2
pin_memory = True
load_model = False
load_model_file = "overfit.pth.tar"
img_dir = "D:\\dataset\\images"
label_dir = "D:\\dataset\\labels"

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes
transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss = loss.item)
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = Yololoss()
    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optimizer)
    
    train_dataset = VOCDataset(
        "D:\\dataset\\8examples.csv",
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    test_dataset = VOCDataset(
        "D:\\dataset\\test.csv",
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )
    for epoch in range(EPOCHS) :
        '''for x, y in train_loader:
            x= x.to(device)
            for idx in range(8):
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4)
                plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
            import sys 
            sys.exit()'''
        #pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        #mean_avg_prec = mean_average_precision(
        #    pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        #)
        #print(f"Train mAP: {mean_avg_prec}")
        train_fn(train_loader, model, optimizer, loss_fn)
    #if mean_avg_prec > 0.9:
    #    checkpoint = {
    #        "state_dict": model.state_dict(),
    #        "optimizer": optimizer.state_dict(),
    #    }
    #    save_checkpoint(checkpoint, filename=load_model_file)
    #    time.sleep(10)
if __name__ == "__main__":
    main()