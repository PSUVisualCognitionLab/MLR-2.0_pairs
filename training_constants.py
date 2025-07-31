training_datasets = {'emnist-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'emnist-skip': {'retina':False, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True},
                     'mnist-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'mnist-skip': {'retina':False, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True},
                     'quickdraw': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'cifar10': {'retina':True, 'rotate':False, 'scale':True},
                     'square': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'line': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'fashion_mnist': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True}}

training_components = {'shape': [['emnist-map', 'mnist-map'], 2], # shape map, weighted 3 times in training etc
                       'color': [['emnist-map', 'mnist-map'], 4], # color map
                       'object': [['quickdraw'], 3], # map for quickdraw
                       'cropped': [['emnist-map', 'mnist-map'], 2], # shape and color recon
                       'cropped_object': [['quickdraw'], 2], # object and color recon
                       'skip_cropped': [['emnist-skip', 'mnist-skip'], 3], # mnist/emnist skip connection
                       'retinal': [['emnist-map', 'mnist-map'], 1], # retinal, scale, location
                       'retinal_object': [['quickdraw'], 1]} # retinal, scale, location, object
