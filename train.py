import os
import sys
import jax
import tqdm
import optax
import argparse
import jax.numpy as jnp
import subprocess as sp
import importlib.util
from flax.training import train_state
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

    # Load train dataet
    train = DataGenerator(data_root=config.train_data_root,
                          mode='train',
                          class_mode=config.class_mode,
                          batch_size=config.batch_size,
                          shuffle=config.shuffle,
                          img_size = config.img_size,
                          color_mode=config.color_mode)
    train_ds = train.get_data()
    
    # Load val dataset
    val = DataGenerator(data_root=config.val_data_root,
                        mode='val',
                        class_mode=config.class_mode,
                        batch_size=config.batch_size,
                        shuffle=config.shuffle,
                        img_size = config.img_size,
                        color_mode=config.color_mode)
    val_ds = val.get_data()
    
    # Get num_classes from dataset
    num_classes = train_ds.num_classes

    def create_train_state(rng, model):
        # Dummy Input for initializing the model
        dummy_input = jnp.ones(
            shape=(config.batch_size, config.img_size[0], config.img_size[1], config.n_channels)
        )
        # Initialize the parameters
        params = model.init(rng, dummy_input)
        model.apply(params, dummy_input)
        # Check the parameters
        jax.tree_map(lambda x: x.shape, params)
        # Create the optimizer
        optimizer = optax.sgd(learning_rate=config.learning_rate, nesterov=config.momentum)
        state = train_state.TrainState.create(apply_fn=model.apply,
                                              tx=optimizer,
                                              params=params)
        return state

    # Define the training step with @jax.jit for faster training
    @jax.jit
    def train_step(state: train_state.TrainState, batch: jnp.ndarray):
        def loss_fn(state, params, batch):
            image, label = batch
            logits = state.apply_fn(
                params,
                image,
                training=True,
                mutable=["batch_stats"],        # for batch normalization
                rngs={"dropout": dropout_rng},  # for dropout
            )[0]
            # print(logits.shape)
            loss = categorical_cross_entropy_loss(logits=logits,
                                                  one_hot_encoded_labels=label)
            return loss, logits

        # Create Gradient Function by passing in the function
        gradient_fn = jax.value_and_grad(
            loss_fn,  # Function to calculate the loss
            argnums=1,  # Parameters are second argument of the function
            has_aux=True,  # Function has additional outputs, here accuracy
        )
        # Pass in the params from the TrainState
        (loss, logits), grads = gradient_fn(state, state.params, batch)
        # Update Parameters
        state = state.apply_gradients(grads=grads)
        return state

    @jax.jit
    def eval_step(state, batch, dropout_rng):
        image, label = batch
        logits = state.apply_fn(
            state.params,
            image,
            training=True,
            mutable=["batch_stats"],        # for batch normalization
            rngs={"dropout": dropout_rng},  # for dropout
        )[0]
        return categorical_metrics(logits=logits, labels=label)
    
    print("Initializing the model")
    # PRNG Key
    rng = jax.random.PRNGKey(config.seed)
    dropout_rng = jax.random.PRNGKey(config.seed)
    # Instantiate the Model
    model = VGG16(num_classes=num_classes)
    # Create the train state
    state = create_train_state(rng, model)
    
    # Start training loop
    best_acc = 0.0
    print("Start training...")
    for epoch in range(1, config.num_epochs + 1):
        # Updates indexes after each epoch
        train_ds.on_epoch_end()
        val_ds.on_epoch_end()
        print(f"Epoch {epoch}/{config.num_epochs}:")
        # For Training
        for step in tqdm.trange(1, len(train_ds)+1, desc="\t \033[94mTraining: "):
            batch_train = train_ds.next()
            state = train_step(state, batch_train)
            train_metrics = eval_step(state, batch_train, dropout_rng)
        print(f"\t Loss: {train_metrics['loss']}, accuracy: {train_metrics['accuracy']}")
        
        # For Validation
        for step in tqdm.trange(1, len(val_ds)+1, desc="\t \033[94mValidation: "):
            batch_val = val_ds.next()
            val_metrics = eval_step(state, batch_val, dropout_rng)
        print(f"\t Loss: {val_metrics['loss']}, accuracy: {val_metrics['accuracy']}")
        
        # Save best val_accuracy
        if val_metrics['accuracy'] > best_acc:
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
