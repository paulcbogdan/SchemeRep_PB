import os
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as utils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse

# NOTE: doesn't work on Python 3.11

def visualize_activation_maps(input, model):
    I = utils.make_grid(input, nrow=1, normalize=True, scale_each=True)
    img = I.permute((1, 2, 0)).cpu().numpy()

    conv_results = []
    x = input
    for idx, operation in enumerate(model.features):
        x = operation(x)
        if idx in {1, 4, 7, 9, 11}:
            conv_results.append(x)

    for i in range(5):
        conv_result = conv_results[i]
        N, C, H, W = conv_result.size()

        mean_acti_map = torch.mean(conv_result, 1, True)
        mean_acti_map = F.interpolate(mean_acti_map, size=[224, 224], mode='bilinear', align_corners=False)

        map_grid = utils.make_grid(mean_acti_map, nrow=1, normalize=True, scale_each=True)
        map_grid = map_grid.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
        map_grid = cv2.applyColorMap(map_grid, cv2.COLORMAP_JET)
        map_grid = cv2.cvtColor(map_grid, cv2.COLOR_BGR2RGB)
        map_grid = np.float32(map_grid) / 255

        visual_acti_map = 0.6 * img + 0.4 * map_grid
        tensor_visual_acti_map = torch.from_numpy(visual_acti_map).permute(2, 0, 1)

        file_name_visual_acti_map = 'conv{}_activation_map.jpg'.format(i + 1)
        utils.save_image(tensor_visual_acti_map, file_name_visual_acti_map)
    return 0

def get_VGG_text(x, model):
    x = batch_img
    for i in range(len(model.features)):
        x = model.features[i](x)
        print(f'{i}: {type(model.features[i])} | {x.shape}')
    print('-' * 50)
    x = model.avgpool(x)
    print(f'avg pool: {type(model.avgpool)} | {x.shape}')
    x = torch.flatten(x, 1)
    print(f'flatten: {type(model.avgpool)} | {x.shape}')
    for i in range(len(model.classifier)):
        x = model.classifier[i](x)
        print(f'{i}: {type(model.classifier[i])} | {x.shape}')
    g = model.forward(batch_img)
    print(f'meh: {g.shape}')


# The first row represents regions where memory was predicted by early visual (layer 2 from VGG16) information, the second row corresponds to middle visual (layer 12), and the last row to late visual (layer 22) information.
# We based the early visual RDM in an early input layer, the middle visual RDM in a middle convolutional layer (CV11), and the late visual RDM in the final fully connected layer (FC2).

if __name__ == '__main__':
    alexnet = models.alexnet(pretrained=True)
    # put the model to eval mode for testing
    alexnet.eval()
    # print(alexnet)
    # quit()

    dir_objs = r'../SchemRep_tasks/PTBtasks/updatedObjectsResampled'
    fns_objs = os.listdir(dir_objs)
    fps_objs = [os.path.join(dir_objs, fn) for fn in fns_objs if '.jpg' in fn]

    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),             # resize the input to 224x224
        transforms.ToTensor(),              # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])

    img = Image.open(fps_objs[0])
    transformed_img = data_transforms(img)
    batch_img = torch.unsqueeze(transformed_img, 0)

    x = alexnet.features[0](batch_img)
    x = alexnet.features[1](x)
    print(f'{x.shape=}')
    print(alexnet.features[1])
    quit()


    output = alexnet(batch_img)
    print("output vector's shape: " + str(output.shape))

    visualize_activation_maps(batch_img, alexnet)



    # https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301