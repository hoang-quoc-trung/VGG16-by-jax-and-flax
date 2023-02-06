seed = 0 # Random seed for PRNG Key
batch_size = 32

# Dataset config
img_size = (224, 224)
n_channels = 3
shuffle = True
train_data_root = '/home/tpnam/trung/cnn/data/ball_dataset/train'
val_data_root = '/home/tpnam/trung/cnn/data/ball_dataset/valid'
test_data_root = '/home/tpnam/trung/cnn/data/ball_dataset/test'
class_mode='categorical'   # 'binary' or 'categorical'
color_mode = 'rgb'

# Training config
learning_rate = 0.001
num_epochs = 1
momentum = None
warmup_epochs = 0
save_best_checkpoint = False
learning_rate_schedule = False

# Checkpoint dir
ckpt_dir = '/home/tpnam/trung/cnn/src/checkpoints/ckpts'