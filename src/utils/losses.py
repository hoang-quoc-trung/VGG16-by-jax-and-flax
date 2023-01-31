import jax
import jax.numpy as jnp
import optax


def categorical_cross_entropy_loss(logits, one_hot_encoded_labels):
    return optax.softmax_cross_entropy(logits=logits,
                                       labels=one_hot_encoded_labels).mean()
    
def sparse_categorical_cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits,
                                                           labels=labels).mean()
    
def binary_cross_entropy_loss(logits,one_hot_encoded_labels):
    return optax.sigmoid_binary_cross_entropy(logits=logits,
                                              labels=one_hot_encoded_labels).mean()

def compute_metrics(logits, labels):
    loss = categorical_cross_entropy_loss(logits=logits, one_hot_encoded_labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics