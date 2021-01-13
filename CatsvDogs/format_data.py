import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

REBUILD_DATA = True # set to true to one once, then back to false unless you want to change something in your training data.

class DogsVSCats():
    CATS = "cats_dogs_imgs\\Cat"
    DOGS = "cats_dogs_imgs\\Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    dirname = os.path.dirname(__file__)

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(os.path.join(self.dirname, label))):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (50,50))
                        img = np.array(img)
                        #img.reshape(3, 50, 50)
                        self.training_data.append([img, np.eye(2)[self.LABELS[label]]])
                    except Exception as e:
                        pass

        
        
        np.random.shuffle(self.training_data)
        data = np.array(self.training_data, dtype=object)
        np.save("training_data.npy", data)

if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

a = np.load('training_data.npy', allow_pickle=True)
print(a[0])