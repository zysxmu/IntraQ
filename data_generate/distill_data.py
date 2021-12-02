
import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from utils import *
import numpy as np
import torch.nn.functional as F
import random
import pickle
import sys
import torchvision.transforms as transforms


class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None


class DistillData(object):
    def __init__(self):
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)
        eps = 1e-6

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def modify_labels(self, labels, targetPro, model_name):

        new_labels = []
        for i in range(len(labels)):

            if random.random() < 0.5:

                label = labels[i]

                if model_name == 'resnet20_cifar10':
                    # label_other = random.randint(0, 9)
                    label_target = torch.FloatTensor(1, 1).uniform_(targetPro, 1).item()
                    label_i = F.one_hot(label, num_classes=10).float()
                elif model_name == 'resnet20_cifar100':
                    # label_other = random.randint(0, 99)
                    label_target = torch.FloatTensor(1, 1).uniform_(targetPro, 1).item()
                    label_i = F.one_hot(label, num_classes=100).float()
                else:
                    # label_other = random.randint(0, 999)
                    label_target = torch.FloatTensor(1, 1).uniform_(targetPro, 1).item()
                    label_i = F.one_hot(label, num_classes=1000).float()

                label_i[label] = label_target
                # label_i[label_other] += 1-label_target
                label_i = label_i.cuda()
                assert torch.argmax(label_i) == label
            else:
                if model_name == 'resnet20_cifar10':
                    label_i = F.one_hot(labels[i], num_classes=10).float()
                elif model_name == 'resnet20_cifar100':
                    label_i = F.one_hot(labels[i], num_classes=100).float()
                else:
                    label_i = F.one_hot(labels[i], num_classes=1000).float()
                label_i = label_i.cuda()

            new_labels.append(label_i)

        new_labels = torch.stack(new_labels, dim=0)
        # import IPython
        # IPython.embed()
        return new_labels.cuda()

    def get_old_data_feature(self, data_path, label_path, teacher_model, hooks):

        data_path_head = data_path[:-8]
        label_path_head = label_path[:-8]

        old_data, old_label = None, None
        for i in range(1, 5):
            # data_path = data_path_head+str(i)+".pickle"

            data_path = data_path_head + str(i) + ".pickle"
            if not os.path.exists(data_path):
                continue
            print(data_path)
            with open(data_path, "rb") as fp:  # Pickling
                gaussian_data_old = pickle.load(fp)
            if old_data is None:
                old_data = np.concatenate(gaussian_data_old, axis=0)
            else:
                old_data = np.concatenate((old_data, np.concatenate(gaussian_data_old, axis=0)))

            # label_path = label_path_head+str(i)+".pickle"
            label_path = label_path_head + str(i) + ".pickle"
            print(label_path)
            with open(label_path, "rb") as fp:  # Pickling
                labels_list_old = pickle.load(fp)
            if old_label is None:
                old_label = np.concatenate(labels_list_old, axis=0)
            else:
                old_label = np.concatenate((old_label, np.concatenate(labels_list_old, axis=0)))
        if old_data is not None:
            print('old_data.shape', old_data.shape, 'old_label', old_label.shape)
        else:
            return {}
        last_feature_dict = {}
        with torch.no_grad():
            teacher_model.eval()
            bs = 64

            for i in range(len(old_data) // bs):
                fake_images, labels = old_data[i * bs:(i + 1) * bs], old_label[i * bs:(i + 1) * bs]
                inp = torch.from_numpy(fake_images).cuda()
                inp_labels = torch.from_numpy(labels).cuda()
                for hook in hooks:
                    hook.clear()
                output = teacher_model(inp)
                last_features = hook.outputs
                gt = inp_labels.data.cpu().numpy()

                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                print(d_acc)

                for j in range(len(gt)):
                    l = gt[j]
                    last_feature = last_features[j]
                    if l not in last_feature_dict:
                        last_feature_dict[l] = []
                    last_feature_dict[l].append(copy.deepcopy(torch.squeeze(last_feature).data.cpu().numpy()))
        return last_feature_dict

    def add_current_data_feature(self, teacher_model, last_feature_dict, hooks, old_data, old_label):

        with torch.no_grad():
            teacher_model.eval()
            bs = 64

            for i in range(len(old_data) // bs):
                fake_images, labels = old_data[i * bs:(i + 1) * bs], old_label[i * bs:(i + 1) * bs]
                inp = torch.from_numpy(fake_images).cuda()
                inp_labels = torch.from_numpy(labels).cuda()
                for hook in hooks:
                    hook.clear()
                output = teacher_model(inp)
                last_features = hook.outputs
                gt = inp_labels.data.cpu().numpy()

                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                print(d_acc)

                for j in range(len(gt)):
                    l = gt[j]
                    last_feature = last_features[j]
                    if l not in last_feature_dict:
                        last_feature_dict[l] = []
                    last_feature_dict[l].append(copy.deepcopy(torch.squeeze(last_feature).data.cpu().numpy()))
        return last_feature_dict


    def getDistilData_hardsample_cosineDistanceEMA_interClass_aug(self,
                                                                  model_name="resnet18",
                                                                  teacher_model=None,
                                                                  batch_size=256,
                                                                  num_batch=1,
                                                                  group=1,
                                                                  targetPro=1.0,
                                                                  cosineMargin=0.4,
                                                                  cosineMargin_upper=0.4,
                                                                  augMargin=0.4,
                                                                  save_path_head=""
                                                                  ):

        data_path = os.path.join(save_path_head, model_name+"refined_gaussian_getDistilData_hardsample_twolabel_" \
                    + str(targetPro) + "cosineMargin"+ str(cosineMargin) + str(cosineMargin_upper)\
                    + "interClassMargin0.0" + "augRHFRRC"+str(augMargin)+ \
                    "_EMAMSE_doubleOptiEpo1000_patience=50OnlyLOR" + str(group) + ".pickle")
        label_path = os.path.join(save_path_head, model_name+"labels_list_getDistilData_hardsample_twolabel_" \
                     + str(targetPro) + "cosineMargin"+ str(cosineMargin) + str(cosineMargin_upper)\
                     + "interClassMargin0.0" + "augRHFRRC"+str(augMargin)+\
                     "_EMAMSE_doubleOptiEpo1000_patience=50OnlyLOR" + str(group) + ".pickle")


        print(data_path, label_path)

        if model_name == 'resnet20_cifar10':
            shape = (batch_size, 3, 32, 32)
        elif model_name == 'resnet20_cifar100':
            shape = (batch_size, 3, 32, 32)
        else:
            shape = (batch_size, 3, 224, 224)


        # initialize hooks and single-precision model
        teacher_model = teacher_model.cuda()
        teacher_model = teacher_model.eval()

        refined_gaussian = []
        labels_list = []

        crit_kl = nn.KLDivLoss(reduction='none').cuda()
        MSE_loss = nn.MSELoss().cuda()
        MSE_loss1 = nn.MSELoss(reduction='none').cuda()
        CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
        CosineEmbeddingLoss = nn.CosineEmbeddingLoss(margin=cosineMargin, reduction='none').cuda()

        hooks, hook_handles = [], []
        for n, m in teacher_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)
            if 'final_pool' in n:
                hook = output_hook()
                hooks.append(hook)
                hook_handles.append(m.register_forward_hook(hook.hook))

        last_feature_dict = self.get_old_data_feature(data_path, label_path, teacher_model, hooks)

        assert batch_size * 5 >= 1000
        # for i, gaussian_data in enumerate(dataloader):
        for i in range((10000//batch_size)+1):
            if i == 5:
                break
            # initialize the criterion, optimizer, and scheduler

            if model_name == 'resnet20_cifar10':
                RRC = transforms.RandomResizedCrop(size=32,scale=(augMargin, 1.0))
            elif model_name == 'resnet20_cifar100':
                RRC = transforms.RandomResizedCrop(size=32,scale=(augMargin, 1.0))
            else:
                RRC = transforms.RandomResizedCrop(size=224,scale=(augMargin, 1.0))

            RHF = transforms.RandomHorizontalFlip()

            gaussian_data = torch.randn(shape).cuda()
            gaussian_data.requires_grad = True
            optimizer = optim.Adam([gaussian_data], lr=0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             min_lr=1e-4,
                                                             verbose=False,
                                                             patience=50)

            if model_name == 'resnet20_cifar10':
                labels = torch.randint(0, 10, (len(gaussian_data),)).cuda()
                labels_mask = F.one_hot(labels, num_classes=10).float()
            elif model_name == 'resnet20_cifar100':
                labels = torch.randint(0, 100, (len(gaussian_data),)).cuda()
                labels_mask = F.one_hot(labels, num_classes=100).float()
            else:
                labels = torch.randint(0, 1000, (len(gaussian_data),)).cuda()
                labels_mask = F.one_hot(labels, num_classes=1000).float()

            new_labels = self.modify_labels(labels, targetPro, model_name)
            gt = labels.data.cpu().numpy()

            old_features = []
            mask = []
            all_sub = []
            for j in range(len(gt)):
                l = gt[j]
                if l not in last_feature_dict:
                    # ll = list(last_feature_dict.keys())[0]
                    if model_name == 'resnet18':
                        old_features.append(torch.randn(512))
                    elif model_name == 'mobilenet_w1':
                        old_features.append(torch.randn(1024))
                    elif model_name == 'mobilenetv2_w1':
                        old_features.append(torch.randn(1280))
                    elif model_name in ['resnet20_cifar10', "resnet20_cifar100"]:
                        old_features.append(torch.randn(64))
                    mask.append(torch.tensor(0.))
                else:
                    # random_index = random.randint(0, len(last_feature_dict[l]) - 1)
                    random_index = 0
                    old_features.append(torch.from_numpy(np.mean(np.stack(last_feature_dict[l]), axis=0)))
                    mask.append(torch.tensor(1.))
                all_sub.append(torch.tensor(-1.))
            old_features = torch.stack(old_features).cuda()
            mask = torch.stack(mask).cuda()
            all_sub = torch.stack(all_sub).cuda()
            print('number of 1 in mask', torch.sum(mask), 'old_features.shape', old_features.shape)

            old_other_features = []
            other_class_features_dict = {}
            other_mask = []
            for j in range(len(gt)):
                l = gt[j]
                if len(last_feature_dict) == 0:
                    old_other_features.append(torch.randn(512))
                    other_mask.append(torch.tensor(0.))
                    continue

                if l not in other_class_features_dict:
                    random_index = random.randint(0, 999)
                    while random_index not in last_feature_dict or last_feature_dict == l:
                        random_index = random.randint(0, 999)
                    other_class_features_dict[l] =  random_index

                old_other_features.append(torch.from_numpy(
                    np.mean(np.stack(last_feature_dict[other_class_features_dict[l]]),
                                                             axis=0)))
                other_mask.append(torch.tensor(1.))

            old_other_features = torch.stack(old_other_features).cuda()
            other_mask = torch.stack(other_mask).cuda()
            print('number of 1 in other_mask', torch.sum(other_mask),
                  'old_other_features.shape', old_other_features.shape)

            for it in range(500*2):

                if random.random() < 0.5:

                    new_gaussian_data = []
                    for j in range(len(gaussian_data)):
                        new_gaussian_data.append(RHF(RRC(gaussian_data[j])))
                    # import IPython
                    # IPython.embed()
                    new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                else:
                    new_gaussian_data = []
                    for j in range(len(gaussian_data)):
                        new_gaussian_data.append(gaussian_data[j])
                    new_gaussian_data = torch.stack(new_gaussian_data).cuda()

                self.mean_list.clear()
                self.var_list.clear()
                for hook in hooks:
                    hook.clear()
                self.teacher_running_mean.clear()
                self.teacher_running_var.clear()
                output = teacher_model(new_gaussian_data)
                last_features = hook.outputs
                last_features = torch.squeeze(last_features)

                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                a = F.softmax(output, dim=1)
                # import IPython
                # IPython.embed()
                loss_target = MSE_loss1(a, new_labels)
                loss_target = loss_target * labels_mask
                loss_target = torch.mean(torch.sum(loss_target, dim=1))

                loss_cosineDistance = torch.sum(
                	torch.clamp(
                        cosineMargin - (1.0 - CosineSimilarity(last_features, old_features.detach())),
                        min=0) * mask) / (torch.sum(mask)+1e-6)

                loss_cosineDistance_upper = torch.sum(
                    torch.clamp(
                        (1.0 - CosineSimilarity(last_features, old_features.detach())) - cosineMargin_upper,
                        min=0) * mask) / (torch.sum(mask) + 1e-6)

                mean_loss = torch.zeros(1).cuda()
                var_loss = torch.zeros(1).cuda()
                for num in range(len(self.mean_list)):
                    mean_loss += MSE_loss(self.mean_list[num], self.teacher_running_mean[num].detach())
                    var_loss += MSE_loss(self.var_list[num], self.teacher_running_var[num].detach())

                mean_loss = mean_loss / len(self.mean_list)
                var_loss = var_loss / len(self.mean_list)

                total_loss = mean_loss + var_loss + loss_target + \
                             loss_cosineDistance + loss_cosineDistance_upper
                print(i, it, 'lr', optimizer.state_dict()['param_groups'][0]['lr'],
                      'd_acc', d_acc, 'mean_loss', mean_loss.item(), 'var_loss',
                      var_loss.item(), 'loss_target', loss_target.item(),
                      'loss_cosineDistance', loss_cosineDistance.item(),
                      'loss_cosineDistance_upper', loss_cosineDistance_upper.item())

                optimizer.zero_grad()
                # update the distilled data
                total_loss.backward()
                optimizer.step()
                scheduler.step(total_loss.item())

            with torch.no_grad():
                output = teacher_model(gaussian_data.detach())
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                print('d_acc', d_acc)

            refined_gaussian.append(gaussian_data.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

            # add current data
            last_feature_dict = self.add_current_data_feature(teacher_model, last_feature_dict, hooks,
                                                                     gaussian_data.detach().cpu().numpy(),
                                                                     labels.detach().cpu().numpy())

            gaussian_data = gaussian_data.cpu()
            del gaussian_data
            del optimizer
            del scheduler
            del labels
            torch.cuda.empty_cache()

        with open(data_path, "wb") as fp:  # Pickling
            pickle.dump(refined_gaussian, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(label_path, "wb") as fp:  # Pickling
            pickle.dump(labels_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()
        # return refined_gaussian


