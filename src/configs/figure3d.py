seed = 0 # seed for PRNG Key
batch_size = 8
num_classes = 30

# Dataset config
img_size = (224, 224)
n_channels = 3
shuffle = True
train_data_root = '/home/tpnam/trung/cnn/data/ball_dataset/train'
class_mode='categorical'

# Training config
learning_rate = 1e-2
num_epochs = 1