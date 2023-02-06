import os
import sys
import tqdm
import optax
import argparse
import jax.numpy as jnp
import subprocess as sp
import importlib.util
import flax
import jax
from flax.training import train_state
from flax.training import lr_schedule
from src.models.model import VGG16
from src.utils.dataloader import DataGenerator
from src.utils.losses import categorical_cross_entropy_loss, sparse_categorical_cross_entropy_loss, binary_cross_entropy_loss
from src.utils.losses import binary_metrics, categorical_metrics
from src.checkpoints.checkpoint import save_checkpoint, load_checkpoint


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def main(args):
    # Set GPU
    print(f"Using GPU {args.gpu_id}")
    os.environ["VISIBLE_CUDA_DEVICES"] = str(args.gpu_id)
    if args.set_memory_growth:
        # Set memory growth for GPU
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        # Set max GPU memory usage
        total_gpu_memory = get_gpu_memory()[args.gpu_id]
        memory_limit = total_gpu_memory * args.gpu_memory_fraction / total_gpu_memory
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "{:.2f}".format(memory_limit)

        # Load the config file
        spec = importlib.util.spec_from_file_location("config", args.config_path)
        config = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config
        spec.loader.exec_module(config)
    
    # Define the training step with @jax.jit for faster training
    @jax.jit
    def train_step(state: train_state.TrainState, image, label, rng):
        def loss_fn(state, params, image, label):
            logits = state.apply_fn(
                params,
                image,
                training=True,
                mutable=["batch_stats"],  # For batch normalization
                rngs={"dropout": rng},    # For dropout
            )[0]
            loss = categorical_cross_entropy_loss(logits=logits, one_hot_encoded_labels=label)
            return loss, logits
        # Create Gradient Function by passing in the function
        gradient_fn = jax.value_and_grad(
            loss_fn,       # Function to calculate the loss
            argnums=1,     # Choose the parameter 'params' in 'loss_fn(state, params, image, label)' 
            has_aux=True,  # Function has additional outputs, here accuracy
        )
        # Pass in the params from the TrainState
        (loss, logits), grads = gradient_fn(state, state.params, image, label)
        # Averaging grads across multiple devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        # Update Parameters
        new_state = state.apply_gradients(grads=grads)
        metrics = binary_metrics(logits=logits, labels=label)
        return new_state, metrics

    # Evaluate for model
    @jax.jit
    def eval_step(state, image, label):
        logits = state.apply_fn(
            state.params,
            image,
            training=False,
            mutable=False
        )[0]
        return binary_metrics(logits=logits, labels=label)
    
    # Create a train state for a model
    def create_train_state(init_rngs, model, learning_rate):
        # Dummy Input for initializing the model
        dummy_input = jnp.ones(
            shape=(1, config.img_size[0], config.img_size[1], config.n_channels)
        )
        # Initialize the parameters
        params = model.init(init_rngs, dummy_input)
        # Create learning rate schedule
        if config.learning_rate_schedule is True:
            steps_per_epoch = int(len(train_ds))
            learning_rate = lr_schedule.create_cosine_learning_rate_schedule(
                config.learning_rate,
                steps_per_epoch,
                config.num_epochs - config.warmup_epochs,
                config.warmup_epochs
            )
        # Create the optimizer
        optimizer = optax.sgd(learning_rate=learning_rate)
        # Create the train state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            tx=optimizer,
            params=params
        )
        return state
    
    # ----------------------------------- Data Loader -----------------------------------
    
    # Get the number of devices (GPU, TPU, CPU...)
    num_devices = jax.device_count()
    print("Number devices:", num_devices)
    # Set batch_size based on number of devices
    batch_size = config.batch_size*num_devices
    # Load train dataset
    train = DataGenerator(data_root=config.train_data_root,
                          mode='train',
                          class_mode=config.class_mode,
                          batch_size=batch_size,
                          shuffle=config.shuffle,
                          img_size = config.img_size,
                          color_mode=config.color_mode)
    train_ds = train.get_data()
    # Get num_classes from dataset
    num_classes = train_ds.num_classes
    # Load val dataset
    val = DataGenerator(data_root=config.val_data_root,
                        mode='val',
                        class_mode=config.class_mode,
                        batch_size=batch_size,
                        shuffle=config.shuffle,
                        img_size = config.img_size,
                        color_mode=config.color_mode)
    val_ds = val.get_data()
    
    # ------------------------------- Initializing The Mode -------------------------------

    # PRNG Key
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, init_rng_dropout = jax.random.split(rng, num=3)
    init_rngs = {'params': init_rng, 'dropout': init_rng_dropout}
    # Instantiate the Model
    model = VGG16(num_classes=num_classes)
    # Create the train state
    state = create_train_state(init_rngs, model, config.learning_rate)
    # Replicate the state so that each device (e.g. GPU) has its own copy
    state = flax.jax_utils.replicate(state)
    # Parallelize the train_step function along the batch axis
    p_train_step = jax.pmap(train_step, axis_name='batch')
    # Parallelize the eval_step function along the batch axis
    p_eval_step = jax.pmap(eval_step, axis_name='batch')
    
    # -------------------------------- Start Training Loop --------------------------------
    
    best_acc = 0.0
    train_batch_metrics = []
    val_batch_metric = []
    print("Start training...")
    for epoch in range(1, config.num_epochs + 1):
        # Reset the first data position
        train_ds.reset()
        val_ds.reset()
        # Reset indexes after each epoch
        train_ds.on_epoch_end()
        val_ds.on_epoch_end()
        print(f"Epoch {epoch}/{config.num_epochs}:")
        # For Training
        for step in tqdm.trange(1, len(train_ds)-1, desc="\t \033[94mTraining: "):
            batch_train = train_ds.next()
            images, labels = batch_train
            """ Reshape images from [num_devices * batch_size, height, width, channels]
                to [num_devices, batch_size, height, width, img_channels]
            """
            image = jnp.reshape(images, (num_devices, -1) + images.shape[1:])
            label = jnp.reshape(labels, (num_devices, -1) + labels.shape[1:])
            # rngs for multi-device
            rng, _ = jax.random.split(rng)
            rngs = jax.random.split(rng, num=num_devices)
            state, train_metrics = p_train_step(state, image, label, rngs)
            train_batch_metrics.append(train_metrics)
        # Compute mean of metrics across each batch in epoch.
        train_batch_metrics = jax.device_get(train_batch_metrics)
        train_epoch_metrics = {
            k: jnp.mean([metrics[k] for metrics in train_batch_metrics])
            for k in train_batch_metrics[0]
        }
        print(f"\t Loss: {train_epoch_metrics['loss']}, accuracy: {train_epoch_metrics['accuracy']*100}%")
        
        # For Validation
        for step in tqdm.trange(1, len(val_ds)-1, desc="\t \033[94mValidation: "):
            batch_val = val_ds.next()
            images, labels = batch_val
            """ Reshape images from [num_devices * batch_size, height, width, channels]
                to [num_devices, batch_size, height, width, img_channels]
            """
            image = jnp.reshape(images, (num_devices, -1) + images.shape[1:])
            label = jnp.reshape(labels, (num_devices, -1) + labels.shape[1:])
            val_metrics = p_eval_step(state, image, label)
            val_batch_metric.append(val_metrics)
        # Compute mean of metrics across each batch in epoch.
        val_batch_metric = jax.device_get(val_batch_metric)
        val_epoch_metrics = {
            k: jnp.mean([metrics[k] for metrics in val_batch_metric])
            for k in val_batch_metric[0]
        }
        print(f"\t Loss: {val_epoch_metrics['loss']}, accuracy: {val_epoch_metrics['accuracy']*100}%")
        
        # Save best val_accuracy
        if val_metrics['accuracy'] > best_acc and config.save_best_checkpoint is True:
            best_acc = val_metrics['accuracy']
            save_checkpoint(state, config.ckpt_dir)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="src/configs/figure3d.py")
    parser.add_argument("--set-memory-growth", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    main(args)
