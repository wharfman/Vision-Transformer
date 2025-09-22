import matplotlib.pyplot as plt
import re

def parse_training_log(filename):
    epochs = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_epoch = None
    
    for line in lines:
        epoch_match = re.match(r'.*Epoch (\d+)/\d+', line)

        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)
        
        train_match = re.search(r'Train Loss: ([\d.]+), Train Accuracy: ([\d.]+)', line)

        if train_match:
            train_losses.append(float(train_match.group(1)))
            train_accuracies.append(float(train_match.group(2)))
        
        val_match = re.search(r'Val Loss: ([\d.]+), Val Accuracy: ([\d.]+)', line)

        if val_match:
            val_losses.append(float(val_match.group(1)))
            val_accuracies.append(float(val_match.group(2)))
        
        lr_match = re.search(r'Learning Rate: ([\d.]+)', line)

        if lr_match:
            learning_rates.append(float(lr_match.group(1)))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates
    }

def plot_metrics(data):
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(data['epochs'], data['train_losses'], 'b-', label='Train Loss', linewidth=1.5, alpha=0.8)
    ax1.plot(data['epochs'], data['val_losses'], 'r-', label='Validation Loss', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax1.fill_between(data['epochs'], data['train_losses'], alpha=0.1, color='blue')
    ax1.fill_between(data['epochs'], data['val_losses'], alpha=0.1, color='red')
    
    ax2.plot(data['epochs'], data['train_accuracies'], 'g-', label='Train Accuracy', linewidth=1.5, alpha=0.8)
    ax2.plot(data['epochs'], data['val_accuracies'], 'orange', label='Validation Accuracy', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax2.fill_between(data['epochs'], data['train_accuracies'], alpha=0.1, color='green')
    ax2.fill_between(data['epochs'], data['val_accuracies'], alpha=0.1, color='orange')
    
    plt.tight_layout()
    
    fig.suptitle('Training Progress Over 300 Epochs', fontsize=16, fontweight='bold', y=1.02)
    
    return fig, (ax1, ax2)

def main():
    filename = 'training_log.txt'
    
    try:
        data = parse_training_log(filename)
    
        fig, axes = plot_metrics(data)
        
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"\nPlots saved as 'training_metrics.png'")
        
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        print("Please ensure the training log file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()