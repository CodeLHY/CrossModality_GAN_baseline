import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

    # Define the ResNet18-based Model


class visible_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(visible_net_resnet, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        x = self.visible.layer3(x)
        x = self.visible.layer4(x)
        x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.dropout(x)
        return x


class thermal_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(thermal_net_resnet, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        x = self.thermal.layer3(x)
        x = self.thermal.layer4(x)
        x = self.thermal.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.dropout(x)
        return x

class Dis(nn.Module):
    def __init__(self, dim_feature):
        super(Dis, self).__init__()
        discriminater = []
        discriminater += [nn.Linear(dim_feature,1)]
        discriminater += [nn.LeakyReLU()]
        self.discriminater = nn.Sequential(*discriminater)
    def forward(self, input):
        result = self.discriminater(input)
        return result
class embed_net_withGAN(nn.Module):
    #  就是resnet 再feature：加上一个线性映射，把resnet50的2048维转换为定义的low_dim维，又加上一个bn
    #  forward时，模式0 即直接将两个模态的数据拼接在一起
    def __init__(self, low_dim, class_num, batch_size, drop=0.5, arch='resnet50'):
        """

        :param low_dim:
        :param class_num:
        :param batch_size: 仅需要看一个模态的即可
        :param drop:
        :param arch:
        """
        super(embed_net_withGAN, self).__init__()
        self.batch_size = batch_size
        if arch == 'resnet18':
            self.visible_net = visible_net_resnet(arch=arch)
            self.thermal_net = thermal_net_resnet(arch=arch)
            pool_dim = 512
        elif arch == 'resnet50':
            self.visible_net = visible_net_resnet(arch=arch)
            self.thermal_net = thermal_net_resnet(arch=arch)
            pool_dim = 2048
        self.feature = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout=drop)
        self.l2norm = Normalize(2)
        self.D_RGB = Dis(low_dim)
        self.D_IR = Dis(low_dim)
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_net(x1)
            x2 = self.thermal_net(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_net(x1)
        elif modal == 2:
            x = self.thermal_net(x2)

        feature_all = self.feature(x)
        feature_RGB = feature_all[0:self.batch_size]
        feature_IR = feature_all[self.batch_size:(self.batch_size*2)]
        result_RGB_True = self.D_RGB(feature_RGB)
        result_RGB_False = self.D_IR(feature_RGB)
        result_IR_True = self.D_IR(feature_IR)
        result_IR_False = self.D_RGB(feature_IR)
        out = self.classifier(feature_all)
        if self.training:
            return out, self.l2norm(feature_all), result_RGB_True, result_RGB_False, result_IR_True, result_IR_False
        else:
            return self.l2norm(x), self.l2norm(feature_all)
    def get_target_tensor(self, result_D, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(result_D)
# debug model structure

# net = embed_net(512, 319)
# net.train()
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# x, y  = net(input, input)