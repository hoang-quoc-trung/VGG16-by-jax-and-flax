from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

class DataGenerator:
    def __init__(self,
                data_root: str,
                mode: str,
                class_mode: str,
                rotation_range: int = 20,
                shear_range: float = 0.2,
                zoom_range: list = [0.8, 1.2],
                horizontal_flip: bool = True,
                brightness_range: list = [0.7, 1.3],
                width_shift_range: float = 0.2,
                height_shift_range: float = 0.2,
                batch_size: int = 16,
                img_size: tuple = (224,224),
                shuffle: bool = True,
                color_mode: str = 'rgb'
                ):
       
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_root = data_root
        self.mode = mode
        self.shuffle = shuffle 
        self.class_mode = class_mode
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.color_mode = color_mode
        
    def get_labels_names(self):
        labels_true = list(set(os.listdir(self.data_root)))
        labels_true.sort()
        labels_true = np.array(labels_true)
        return labels_true
    
    def get_data(self):
        if self.mode=='train':
            train_datagen = ImageDataGenerator(rescale = 1./255,
                                               rotation_range=self.rotation_range,
                                               shear_range=self.shear_range,
                                               zoom_range = self.zoom_range,
                                               horizontal_flip=self.horizontal_flip,
                                               brightness_range= self.brightness_range,
                                               width_shift_range=self.width_shift_range,
                                               height_shift_range=self.height_shift_range,
                                               fill_mode='nearest')
            
            data = train_datagen.flow_from_directory(directory=self.data_root,
                                                     target_size=self.img_size,
                                                     batch_size=self.batch_size,
                                                     class_mode=self.class_mode,
                                                     color_mode=self.color_mode,
                                                     shuffle=self.shuffle)
        else:
            datagen = ImageDataGenerator(rescale=1./255)
            data = datagen.flow_from_directory(directory=self.data_root,
                                               target_size=self.img_size,
                                               batch_size=self.batch_size,
                                               class_mode=self.class_mode,
                                               color_mode=self.color_mode,
                                               shuffle=self.shuffle)
        return data
        
