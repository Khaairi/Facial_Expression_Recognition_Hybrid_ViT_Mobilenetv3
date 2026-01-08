import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random
from pathlib import Path

from data_setup import load_and_split_data, create_transforms, create_datasets, create_dataloaders
from model import ViTMobilenet, SAM, EarlyStopping

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
DATA_PATH = project_root / "data" / "fer2013v2_clean.csv"

def get_args():
    parser = argparse.ArgumentParser(description="Train Hybrid ViT-MobileNet with SAM")

    # Experiment / Logging
    parser.add_argument("--model_name", type=str, default="hybrid_mobilenet_vit_pooling_SAM", 
                        help="Name of the model for saving files (checkpoints, plots)")
    parser.add_argument("--save_dir", type=str, default=str(project_root / "results"), 
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloaders")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    
    # Optimization
    parser.add_argument("--sam_rho", type=float, default=0.05, help="Rho parameter for SAM optimizer")
    parser.add_argument("--patience", type=int, default=10, help="Patience for Early Stopping")
    parser.add_argument("--sched_patience", type=int, default=5, help="Patience for LR Scheduler")
    parser.add_argument("--sched_factor", type=float, default=0.1, help="Factor for LR Scheduler")

    # Data
    parser.add_argument("--data_path", type=str, default=str(DATA_PATH), help="Path to CSV data file")
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, args, device):
    # Initialize training utilities
    base_optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.sam_rho)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.sched_factor, patience=args.sched_patience)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0)

    # Define path
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize lists to store training and validation metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Initialize the best metric for model saving
    best_val_accuracy = -float('inf')

    print(f"Starting training for {args.epochs} epochs on {device}...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
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
        print(f"Epoch {epoch + 1}/{args.epochs}: "
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
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} (Validation)")
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
        print(f"Epoch {epoch + 1}/{args.epochs}: "
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
            model_filename = f"{args.model_name}_best.pt"
            model_path = os.path.join(args.save_dir, model_filename)
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
        loss_plot_path = os.path.join(args.save_dir, f"{args.model_name}_loss.png")
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
        accuracy_plot_path = os.path.join(args.save_dir, f"{args.model_name}_accuracy.png")
        plt.savefig(accuracy_plot_path)
        plt.close()
        
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}!")
            break

if __name__ == "__main__":
    args = get_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Configuration: {args}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    train_transforms, test_transforms = create_transforms(args.img_size)
    data_train, data_val, data_test = load_and_split_data(args.data_path, args.seed)
    train_dataset, val_dataset, test_dataset = create_datasets(data_train, data_val, data_test, train_transforms, test_transforms)

    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size, args.seed)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    model = ViTMobilenet(num_classes=len(class_names), 
                in_channels=3, 
                num_heads=12, 
                embedding_dim=768, 
                num_transformer_layers=12,
                mlp_size=3072)
    model.to(DEVICE)
    
    train_model(model, train_loader, val_loader, args, DEVICE)