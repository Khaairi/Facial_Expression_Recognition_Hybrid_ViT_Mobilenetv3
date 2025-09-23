import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random

from dataset import load_and_split_data, create_transforms, create_datasets, create_dataloaders
from model import ViTMobilenet, SAM, EarlyStopping

SEED = 123
DATA_PATH = "data/fer2013v2_clean.csv"
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

def train_model(model, train_loader, val_loader):
    # Initialize training utilities
    base_optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    early_stopping = EarlyStopping(patience=10, min_delta=0)

    # Define path
    SAVE_PATH = "results"
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Initialize lists to store training and validation metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Initialize the best metric for model saving
    best_val_accuracy = -float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)  # Update weights

            # Update statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{train_loss / (batch_idx + 1):.4f}",
                "Acc": f"{correct / total:.4f}"
            })

        # Calculate training accuracy and loss
        train_accuracy = correct / total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Print training summary
        print(f"Epoch {epoch + 1}/{EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_targets = []
        all_predicted = []

        with torch.no_grad():  # Disable gradient computation
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} (Validation)")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Update statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # Collect all targets and predictions for F1-score
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{val_loss / (batch_idx + 1):.4f}",
                    "Acc": f"{val_correct / val_total:.4f}"
                })

        # Calculate validation accuracy, loss, and F1-score
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_targets, all_predicted, average="weighted")
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Print validation summary
        print(f"Epoch {epoch + 1}/{EPOCHS}: "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"Val F1: {val_f1:.4f}")

        # Step the learning rate scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join(SAVE_PATH, "hybrid_mobilenet_vit_pooling_SAM_best.pt")
            torch.save({
                "model_state_dict": model.state_dict()
            }, model_path)
    #         torch.save(model.state_dict(), model_path)
            print(f"Best model saved at {model_path} with val accuracy: {best_val_accuracy:.4f}")

        # Save loss and accuracy plots
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker='o')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='o')
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(SAVE_PATH, "hybrid_mobilenet_vit_pooling_SAM_loss.png")
        plt.savefig(loss_plot_path)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy", marker='o')
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy", marker='o')
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        accuracy_plot_path = os.path.join(SAVE_PATH, "hybrid_mobilenet_vit_pooling_SAM_accuracy.png")
        plt.savefig(accuracy_plot_path)
        plt.close()
        
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}!")
            break

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    train_transforms, test_transforms = create_transforms()
    data_train, data_val, data_test = load_and_split_data(DATA_PATH)
    train_dataset, val_dataset, test_dataset = create_datasets(data_train, data_val, data_test, train_transforms, test_transforms)

    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    model = ViTMobilenet(num_classes=len(class_names), 
                in_channels=3, 
                num_heads=12, 
                embedding_dim=768, 
                num_transformer_layers=12,
                mlp_size=3072)
    model.to(DEVICE)
    
    train_model(model, train_loader, val_loader)