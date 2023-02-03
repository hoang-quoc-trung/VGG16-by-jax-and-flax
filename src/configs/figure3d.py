seed = 0 # Random seed for PRNG Key
batch_size = 8

# Dataset config
img_size = (224, 224)
n_channels = 3
shuffle = True
train_data_root = '/home/tpnam/trung/cnn/data/ball_dataset/train'
val_data_root = '/home/tpnam/trung/cnn/data/ball_dataset/valid'
test_data_root = '/home/tpnam/trung/cnn/data/ball_dataset/test'
class_mode='categorical'
color_mode = 'rgb'

# Training config
learning_rate = 0.001
num_epochs = 10
momentum = None

# Checkpoint dir
ckpt_dir = '/home/tpnam/trung/cnn/src/checkpoints'