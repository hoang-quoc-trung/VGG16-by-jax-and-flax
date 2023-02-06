import jax
import jax.numpy as jnp
import optax


def categorical_cross_entropy_loss(logits, one_hot_encoded_labels):
    """
    It computes the mean of the cross entropy loss between the logits and the one-hot encoded labels
    
    Args:
        logits: The output of the softmax activation layer
        one_hot_encoded_labels: The labels in one-hot encoded format
        return: The mean of the softmax cross entropy loss
    """
    return optax.softmax_cross_entropy(logits=logits,
                                       labels=one_hot_encoded_labels).mean()
    
    
def sparse_categorical_cross_entropy_loss(logits, one_hot_encoded_labels):
    """
    It computes the cross entropy loss between the logits and the one-hot encoded labels
    
    Args:
        logits: The output of the neural network
        one_hot_encoded_labels: The labels in one-hot encoded format
    
    Return: The mean of the softmax cross entropy with integer labels
    """
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits,
                                                           labels=one_hot_encoded_labels).mean()
    
    
def binary_cross_entropy_loss(logits,one_hot_encoded_labels):
    """
    It computes the binary cross entropy loss between the logits and the one-hot encoded labels
    
    Args:
        logits: The output of the sigmoid activation layer
        one_hot_encoded_labels: The labels in one-hot encoded format
    
    Return: The mean of the sigmoid binary cross entropy loss
    """
    return optax.sigmoid_binary_cross_entropy(logits=logits,
                                              labels=one_hot_encoded_labels).mean()


def mean_squared_error(logits, labels):
    """
    It takes two arrays, subtracts them, squares the result, and then takes the mean
    
    Args:
        logits: The output of the model
        labels: The true values of the data
    
    Return: The mean squared error of the logits and labels.
    """
    return jnp.mean((logits - labels)**2)


def categorical_metrics(logits, labels):
    """
    This function calculates the categorical cross entropy loss and accuracy of a classification model.

    Args:
    logits: The output of softmax activation layer
    labels: The true labels in one-hot encoded format

    Returns: A dictionary containing the categorical cross entropy loss and accuracy of the model.
    """
    loss = categorical_cross_entropy_loss(logits=logits, one_hot_encoded_labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def binary_metrics(logits, labels, threshold=0.5):
    """
    It computes the binary cross entropy loss and accuracy metrics between the logits and the labels.

    Args:
        logits: The output of the sigmoid activation layer.
        labels: The true binary labels.
        threshold: The threshold value to convert the logits into binary predictions (0 or 1).

    Returns: A dictionary containing the binary cross entropy loss and accuracy.
    """
    loss = binary_cross_entropy_loss(logits=logits, one_hot_encoded_labels=labels)
    accuracy = jnp.mean(jnp.where(logits > threshold, 1.0, 0.0) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics
