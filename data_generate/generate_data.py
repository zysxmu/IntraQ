
import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'mobilenet_w1',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'resnet20_cifar100', 'regnetx_600m'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--group',
                        type=int,
                        default=1,
                        help='group of generated data')
    parser.add_argument('--targetPro',
                        type=float,
                        default=1.0,
                        help='targetPro')
    parser.add_argument('--cosineMargin',
                        type=float,
                        default=0.4,
                        help='cosineMargin')
    parser.add_argument('--cosineMargin_upper',
                        type=float,
                        default=0.4,
                        help='cosineMargin_upper')
    parser.add_argument('--augMargin',
                        type=float,
                        default=0.4,
                        help='interClassMargin')
    parser.add_argument('--save_path_head',
                        type=str,
                        default='',
                        help='save_path_head')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    if args.model == 'regnetx_600m':
        from models.regnet import regnetx006
        model = regnetx006(pretrained=True)
    else:
        model = ptcv_get_model(args.model, pretrained=True)
    print('****** Full precision model loaded ******')

    # # Load validation data
    # test_loader = getTestData(args.dataset,
    #                           batch_size=args.test_batch_size,
    #                           path='/media/disk1/ImageNet2012/',
    #                           for_inception=args.model.startswith('inception'))
    # print('****** Test model! ******')
    # test(model.cuda(), test_loader)
    # Generate distilled data
    DD = DistillData()
    print(args.group, args.targetPro)
    dataloader = DD.getDistilData_hardsample_cosineDistanceEMA_interClass_aug(
        model_name=args.model,
        teacher_model=model.cuda(),
        batch_size=args.batch_size,
        group=args.group,
        targetPro=args.targetPro,
        cosineMargin=args.cosineMargin,
        cosineMargin_upper=args.cosineMargin_upper,
        augMargin=args.augMargin,
        save_path_head=args.save_path_head
    )

    print('****** Data Generated ******')




