import torch
import torch.nn as nn
import torchvision.models as models
import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.utils.utils import print_msg


def generate_model(cfg):
    model = build_model(
        network=cfg.train.network,
        num_classes=cfg.data.num_classes,
        pretrained=cfg.train.pretrained,
        train_head=cfg.train.train_head
    )
    
    if cfg.train.checkpoint:
        checkpoint = torch.load(cfg.train.checkpoint)
        weights = checkpoint['network']
        model.load_state_dict(weights, strict=True)
        print_msg('Load weights form {}'.format(cfg.train.checkpoint))
        
    if 'cuda' in cfg.base.device and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(cfg.base.device)
    return model


def build_model(network: str = 'resnet18',
                num_classes: int = 1000,
                pretrained: bool = False,
                train_head: bool = False):
    
    model = BUILDER_MODEL[network](pretrained=pretrained)
    if train_head:
        set_parameter_requires_grad(model, feature_extract=True)
    
    if 'alexnet' in network or 'vgg' in network:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    elif 'resnet' in network or 'resnext' in network or 'shufflenet_v2' in network or 'wide_resnet' in network:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif 'densenet' in network:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif 'efficientnet' in network:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif 'convnext' in network:
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        
    elif 'mobilenet_v2' in network:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    
    elif 'mobilenet_v3' in network:
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    elif 'squeezenet' in network:
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes

    return model


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


BUILDER_MODEL = {
    'alexnet': models.alexnet,
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'vgg11_bn': models.vgg11_bn,
    'vgg13_bn': models.vgg13_bn,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b5': models.efficientnet_b5,
    'efficientnet_b6': models.efficientnet_b6,
    'efficientnet_b7': models.efficientnet_b7,
    'efficientnet_v2_s': models.efficientnet_v2_s,
    'efficientnet_v2_m': models.efficientnet_v2_m,
    'efficientnet_v2_l': models.efficientnet_v2_l,
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'convnext_base': models.convnext_base,
    'convnext_large': models.convnext_large,
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'resnext101_64x4d': models.resnext101_64x4d,
    'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,
    'squeezenet1_0': models.squeezenet1_0,
    'squeezenet1_1': models.squeezenet1_1,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2,
    # 'vit_b_16': models.vit_b_16,
    # 'vit_b_32': models.vit_b_32,
    # 'vit_l_16': models.vit_l_16,
    # 'vit_l_32': models.vit_l_32,
    # 'vit_h_14': models.vit_h_14,
}

if __name__ == '__main__':
    print("hello world")
