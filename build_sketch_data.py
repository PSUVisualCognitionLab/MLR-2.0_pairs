import numpy as np
import os

# must download the individual class npy bitmaps from: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/binary?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&inv=1&invt=Ab0unA
# save this data into one folder that matches data_dir below

data_dir = 'data/quickdraw_npy'
object_classes = ['airplane', 'bird', 'car', 'cat', 'dog', 'duck', 'frog', 'horse', 'sailboat', 'truck']

all_images = []
all_labels = []

# iterate through each class, load, and append
for label, obj in enumerate(object_classes):
    filename = f'full_numpy_bitmap_{obj}.npy'
    filepath = os.path.join(data_dir, filename)
    print(f'Loading {filepath}...')
    
    if obj == 'car':
        data = np.load(filepath)  # shape: (N, 784)
    
    else:
        data = np.load(filepath)[:70000]
    
    all_images.append(data)
    all_labels.append(np.full(len(data), label, dtype=np.uint8))

# concat images and labels
x = np.concatenate(all_images, axis=0)
y = np.concatenate(all_labels, axis=0)

xy = np.concatenate([x, y[:, None]], axis=1)

# save combined dataset
output_path = os.path.join(data_dir, 'full_numpy_bitmap_all_objs.npy')
np.save(output_path, xy)

print(f'Saved merged dataset to: {output_path}')
print(f'Shape of final dataset: {xy.shape} (images + labels)')