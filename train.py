import os
import sys
import jax
import tqdm
import optax
import argparse
import numpy as np
import jax.numpy as jnp
import subprocess as sp
import importlib.util
from flax.training import train_state
from src.models.model import VGG16
from src.utils.dataloader import DataGenerator
from src.utils.losses import categorical_cross_entropy_loss, compute_metrics


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

    # Load the dataset
    print(f"Loading the dataset from {config.train_data_root}")

    train = DataGenerator(data_root=config.train_data_root,
                          mode='train',
                          class_mode=config.class_mode,
                          batch_size=config.batch_size,
                          shuffle=config.shuffle,
                          img_size = config.img_size)
    train_ds = train.get_data()
    num_classes = train_ds.num_classes

    print("Initializing the model")
    # PRNG Key
    rng = jax.random.PRNGKey(config.seed)
    dropout_rng = jax.random.PRNGKey(config.seed)
    # Dummy Input for initializing the model
    dummy_input = jnp.ones(
        shape=(config.batch_size, config.img_size[0], config.img_size[1], 3)
    )
    # Instantiate the Model
    model = VGG16(num_classes=num_classes)
    # Initialize the parameters
    params = model.init(rng, dummy_input)
    model.apply(params, dummy_input)
    # Check the parameters
    jax.tree_map(lambda x: x.shape, params)

    # Create the optimizer
    optimizer = optax.adam(config.learning_rate)
    # Create the train state
    state = train_state.TrainState.create(
        apply_fn=model.apply, tx=optimizer, params=params
    )

    # Define the training step with @jax.jit for faster training
    @jax.jit
    def train_step(state: train_state.TrainState, batch: jnp.ndarray):
        def loss_fn(state, params, batch):
            image, label = batch
            logits = state.apply_fn(
                params,
                image,
                training=True,
                mutable=["batch_stats"],  # for batch normalization
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

        return state, loss

    @jax.jit
    def eval_step(state, batch):
        image, label = batch
        logits = state.apply_fn(
            state.params,
            image,
            training=True,
            mutable=["batch_stats"],  # for batch normalization
            rngs={"dropout": dropout_rng},  # for dropout
        )[0]
        return compute_metrics(logits=logits, labels=label)

    # Start training loop
    print("Start training...")
    for epoch in range(1, config.num_epochs + 1):
        train_ds.on_epoch_end()
        print(f"Epoch {epoch}/{config.num_epochs}:")
        # Training
        for index in range(1, len(train_ds) + 1):
            batch_train = train_ds.next()
            # Train the model
            state, loss = train_step(state, batch_train)
            metrics = eval_step(state, batch_train)
            print(f"\t [{index}/{len(train_ds) + 1}] Training_Loss: {loss}, Training_accuracy: {metrics['accuracy']}")


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
