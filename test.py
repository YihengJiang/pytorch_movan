#!/usr/bin/env python
# -*- coding:utf-8 -*-
# import torchvision
# import torch.optim as optim
# import torch.nn as nn
# # 局部微调
# # 有时候我们加载了训练模型后，只想调节最后的几层，
# # 其他层不训练。其实不训练也就意味着不进行梯度计算，PyTorch中提供的requires_grad使得对训练的控制变得非常简单。
# model = torchvision.models.resnet18(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# # 替换最后的全连接层， 改为训练100类
# # 新构造的模块的参数默认requires_grad为True
# model.fc = nn.Linear(512, 100)
#
#
# # 只优化最后的分类层
# optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
#
#
#
# ###################################
# #上面和下面的微调方法只能选择一种，
# #  ########################################################
#
#
# # 全局微调
# # 有时候我们需要对全局都进行finetune，只不过我们希望改换过的层和其他层的学习速率不一样，
# # 这时候我们可以把其他层和新层在optimizer中单独赋予不同的学习速率。比如：
# ignored_params = list(map(id, model.fc.parameters()))
# base_params = filter(lambda p: id(p) not in ignored_params,model.parameters())
# #this is the new way to use Optimd
# optimizer = optim.SGD([
#             {'params': base_params},
#             {'params': model.fc.parameters(), 'lr': 1e-3}
#             ], lr=1e-2, momentum=0.9)

import torchvision.models as models
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


class CNN(nn.Module):

	def __init__(self, block, layers, num_classes=9):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		# 新增一个反卷积层
		self.convtranspose1 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=1, output_padding=0,
												 groups=1, bias=False, dilation=1)
		# 新增一个最大池化层
		self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		# 去掉原来的fc层，新增一个fclass层
		self.fclass = nn.Linear(2048, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		# 新加层的forward
		x = x.view(x.size(0), -1)
		x = self.convtranspose1(x)
		x = self.maxpool2(x)
		x = x.view(x.size(0), -1)
		x = self.fclass(x)

		return x

# 加载model


resnet50 = models.resnet50(pretrained=True)
cnn = CNN(Bottleneck, [3, 4, 6, 3])
# 读取参数
pretrained_dict = resnet50.state_dict()
model_dict = cnn.state_dict()
# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
cnn.load_state_dict(model_dict)
# print(resnet50)
print(cnn)
