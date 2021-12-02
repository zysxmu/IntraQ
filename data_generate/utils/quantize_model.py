#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import torch.nn as nn
import copy
from .quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock


def quantize_model(model, bit=8):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d:
        quant_mod = Quant_Conv2d(weight_bit=bit)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=bit)
        quant_mod.set_param(model)
        return quant_mod

    # quantize all the activation to 8-bit
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
        return nn.Sequential(*[model, QuantAct(activation_bit=bit)])

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, bit=bit))
        return nn.Sequential(*mods)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(q_model, attr, quantize_model(mod, bit=bit))
        return q_model


def quantize_model_resnet18(model, bit=None, module_name='model'):
        """
        Recursively quantize a pretrained single-precision model to int8 quantized model
        model: pretrained single-precision model
        """
        weight_bit = bit
        act_bit = bit
        # import IPython
        # IPython.embed()
        # quantize convolutional and linear layers
        if type(model) == nn.Conv2d:
            if module_name == 'model.features.init_block.conv.conv':
                quant_mod = Quant_Conv2d(weight_bit=bit)
                quant_mod.set_param(model)
                return quant_mod
            if bit is not None:
                quant_mod = Quant_Conv2d(weight_bit=bit)
            else:
                quant_mod = Quant_Conv2d(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.Linear:
            # quant_mod = Quant_Linear(weight_bit=weight_bit)
            quant_mod = Quant_Linear(weight_bit=bit)
            quant_mod.set_param(model)
            return quant_mod

        # quantize all the activation
        elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
            # import IPython
            # IPython.embed()
            if module_name == 'model.features.stage4.unit2.activ':
                return nn.Sequential(*[model, QuantAct(activation_bit=bit)])
            if bit is not None:
                return nn.Sequential(*[model, QuantAct(activation_bit=bit)])
            else:
                return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])

        # recursively use the quantized module to replace the single-precision module
        elif type(model) == nn.Sequential:
            mods = []
            for n, m in model.named_children():
                if n == 'init_block':
                    mods.append(quantize_model_resnet18(m, bit, module_name + '.' + n))
                else:
                    mods.append(quantize_model_resnet18(m, bit, module_name + '.' + n))
            return nn.Sequential(*mods)
        else:
            q_model = copy.deepcopy(model)

            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    setattr(q_model, attr, quantize_model_resnet18(mod, bit, module_name + '.' + attr))
            return q_model

def freeze_model(model):
    """
    freeze the activation range
    """
    if type(model) == QuantAct:
        model.fix()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)
        return model


def unfreeze_model(model):
    """
    unfreeze the activation range
    """
    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)
        return model
