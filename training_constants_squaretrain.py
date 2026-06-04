from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms, utils

training_related_pairs = [('airplane', 'bird'), ('tree', 'axe')]
testing_related_pairs = [('airplane', 'bird')]
testing_unrelated_pairs = [('airplane', 'axe')]

# dataset names must be in format <dataset name>-<component type>, unless there is only one component trained by that dataset
training_datasets = {'emnist-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'emnist-skip': {'retina':False, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True},
                     'mnist-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'mnist-skip': {'retina':False, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True},
                     'quickdraw-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'quickdraw-skip': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'skip': True},
                     'cifar10': {'retina':True, 'rotate':False, 'scale':True},
                     'square-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'line': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'fashion_mnist': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'quickdraw_pairs-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'class_pairs': training_related_pairs}}

training_components = {'shape': [['square-map'], 2], # shape map, weighted 3 times in training etc
                       'color': [['square-map'], 3], # color map
                       'object': [['square-map'], 1], # map for quickdraw
                       'cropped': [['square-map'], 2], # shape and color recon
                       'cropped_object': [['square-map'], 1], # object and color recon
                       'skip_cropped': [['emnist-skip'], 2], # mnist/emnist skip connection
                       'retinal': [['square-map'], 1], # retinal, scale, location
                       'retinal_object': [['square-map'], 1]} # retinal, scale, location, object

def text_to_tensor(text,height,width):
    img = Image.new('RGB', (width, height), (255, 255, 255))
    ImageDraw.Draw(img).text((10, 10), text, fill=(0, 0, 0))
    return transforms.ToTensor()(img)