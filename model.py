import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3),
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

class Cnnblock(nn.Module):
    def __init__(self, in_channels, out_channls, **kwargs):
        super(Cnnblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channls, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channls)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    Cnnblock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers.append(nn.MaxPool2d(2, 2))
            elif type(x) ==list:
                conv1 = x[0]
                conv2 = x[1]
                for _ in range(x[2]):
                    layers.append(Cnnblock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))
                    layers.append(Cnnblock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S ,B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(1024 * S *S, 496),
            nn.Dropout(0.0),               #训练完用的时候记得关
            nn.LeakyReLU(0.1),
            nn.Linear( 496, S * S * (C + B * 5))
        )
if __name__ == "__main__":
    x=torch.rand(1, 3, 448, 448)
    net = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    output = net(x)
    print(output.shape) 