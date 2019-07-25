import torch
from torchvision import transforms
import cv2
import numpy as np
from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg16


weight_path1 = "weights/vgg16-397923af.pth"
im_path = "data/car.jpg"

from PIL import Image
ima = Image.open(im_path)

transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize
(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

img_t = transform(ima)

print(img_t.shape)
print("meanof image data",torch.mean(img_t))
print("std of image_data",torch.std(img_t))

batch_t = torch.unsqueeze(img_t, 0)
print("batch_size",batch_t.shape)

tree  = open("imagenet_classes.txt")
list1 = list()
for i in tree.readlines():
    l = i.strip()
    list1.append(l)

print("no of classes",len(list1))

Network = vgg16(False)

Network.load_state_dict(torch.load(weight_path1))
output = Network(batch_t)

#print(output)
_, index = torch.max(output, 1)

value = index[0].item()
print("maimum value in vector is : --",value)

p = torch.nn.functional.softmax(output,dim=1)[0]

#print(p[value]*100)

print("this is --->",list1[value])
print("probability is :",p[value]*100)