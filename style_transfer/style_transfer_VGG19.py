from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

# Get "features" from VGG19 ("classifier" portion isn't needed)
vgg = models.vgg19(pretrained=True).features

# Freeze all VGG params since we're onl optimizing target image
for param in vgg.parameters():
    param.requires_grad_(False)

# move vgg model to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

print(vgg)


def load_image(img_path, max_size=400, shape=None):
    ''' Load and transform and image, making sure it is <= 400 pixels
    in the x-y dims '''
    image = Image.open(img_path).convert('RGB')

    # Large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Discard the transparent, alpha channel (that's the :3) and add batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


# Load content and style image
content = load_image('style_transfer/imgs/me_buda.jpg').to(device)

# Resize style to match content, to make code easier
style = load_image('style_transfer/imgs/dali_memoria.jpg', shape=content.shape[-2:]).to(device)


# Helper function to un-nomarmalize an image and convert it from Tensor image
# to a NumPy image for display
def img_convert(tensor_img):
    '''Display a tensor as image'''

    img = tensor_img.to('cpu').clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0, 1)

    return img


# Display the images if run on console, othewise just comment the code
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# Content and style side-by-side
ax1.imshow(img_convert(content))
ax2.imshow(img_convert(style))


def get_features(image, model, layers=None):
    ''' Run an image forward trough a model and get the features for a set of layers.
    Default layers are for VGGNet matching Gatys et al (2016). '''

    # Mapping PyTorch's VGGNet names to the names from original paper
    # Layers for content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content representation
                  '28': 'conv5_1'}

        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features




''' Calculate the Gram Matrix of given tensor 
    https://en.wikipedia.org/wiki/Gramian_matrix '''


def gram_matrix(tensor):
    # Get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()

    # Reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(d, h * h)

    # Calculate the Gram Matrix
    gram = torch.mm(tensor, tensor.t())

    return gram


# Get content and style features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Calculate the Gram Matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# todo -> watch lectures
# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1  # alpha
style_weight = 1e6  # beta

# for displaying the target image, intermittently
show_every = 400

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 2000  # decide how many iterations to update your image (5000)

for ii in range(1, steps + 1):

    # get the features from your target image
    target_features = get_features(target, vgg)

    # the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    # calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # display intermediate images and print the loss
    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(img_convert(target))
        plt.show()

# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(img_convert(content))
ax2.imshow(img_convert(target))