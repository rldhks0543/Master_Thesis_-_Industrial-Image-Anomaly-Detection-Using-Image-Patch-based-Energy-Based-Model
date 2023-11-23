from PIL import Image
import argparse
import random
import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms
from tqdm import tqdm


class MVTEC(data.Dataset):

# category 상관없이 다 불러오는 코드

    def __init__(self, args, root, train=True,
                 transform=None, target_transform=None,
                 category=['bottle', 'cable', 'capsule', 
                           'carpet', 'grid', 'hazelnut', 
                           'leather', 'metal_nut', 'pill', 
                           'screw', 'tile', 'toothbrush', 
                           'transistor', 'wood', 'zipper'],
                            xy_std_dev = None,
                resize=None, interpolation=2):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.xy_std_dev = xy_std_dev
        self.resize = resize
        self.interpolation = interpolation
        self.args = args

        # ***gray scale image 존재함. grid, screw, zipper

        assert type(category) == type([]), '!!! Category must be list !!!'
        
        # load images for training
        if self.train:
            label = 0
            self.train_data = []
            self.train_labels = []
            for cat in category:
                print(f'(Train)Dataloader : {cat} load start')
                cwd = os.getcwd()
                trainFolder = args.data_root + '/' +  cat + '/train/good'
                os.chdir(trainFolder)
                filenames = [f.name for f in os.scandir()]
                for file in tqdm(filenames):
                    img = mpimg.imread(file)
                    img = img*255
                    img = img.astype(np.uint8)
                    self.train_data.append(img)
                    self.train_labels.append(label)
                os.chdir(cwd)
                label += 1
            self.train_data = np.array(self.train_data)

        # load images for testing
        else:
            self.test_data = []
            self.test_labels = []
            for cat in category:
                print(f'(Test)Dataloader : {cat} load start')
                cwd = os.getcwd()
                testFolder = args.data_root + '/' + cat + '/test/'
                os.chdir(testFolder)
                subfolders = [sf.name for sf in os.scandir() if sf.is_dir()]
                cwsd = os.getcwd()

                # for every subfolder in test folder
                for subfolder in subfolders:
                    label = 0
                    if subfolder == 'good':
                        label = 1
                    testSubfolder = './'+subfolder+'/'
                    os.chdir(testSubfolder)
                    filenames = [f.name for f in os.scandir()]
                    for file in filenames:
                        img = mpimg.imread(file)
                        img = img*255
                        img = img.astype(np.uint8)
                        self.test_data.append(img)
                        self.test_labels.append(label)
                    os.chdir(cwsd)
                os.chdir(cwd)
            self.test_data = np.array(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img).convert("RGB")

            center_x = (img.width - self.args.crop_size) // 2
            center_y = (img.height - self.args.crop_size) // 2
            x_std_dev, y_std_dev = self.xy_std_dev

            left = int(random.gauss(center_x, x_std_dev))
            top = int(random.gauss(center_y, y_std_dev))
            # 범위 보정
            left = max(1, min(left, img.width - self.args.crop_size))
            top = max(1, min(top, img.height - self.args.crop_size))

            right = left + self.args.crop_size
            bottom = top + self.args.crop_size
            img = img.crop((left, top, right, bottom))
            
        else:
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.fromarray(img).convert("RGB")

        #if resizing image
        if self.resize is not None:
            resizeTransf = transforms.Resize(self.resize, self.interpolation)
            img = resizeTransf(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Args:
            None
        Returns:
            int: length of array.
        """
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)