import matplotlib.pyplot as plt

def visualize_metrics(num_epochs, train_accuracy, val_accracy, train_loss, val_loss):
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, train_accuracy, label='Training')
    plt.plot(epochs_range, val_accracy, label='Validation')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, train_loss, label='Training')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout()
    plt.show()
    
    
