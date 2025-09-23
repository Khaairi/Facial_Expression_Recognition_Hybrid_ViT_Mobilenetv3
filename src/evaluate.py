import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset import load_and_split_data, create_transforms, create_datasets, create_dataloaders
from model import ViTMobilenet

SEED = 123
DATA_PATH = "data/fer2013v2_clean.csv"
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/hybrid_mobilenet_vit_pooling_SAM_best.pt"

def evaluate_model(best_model, test_loader):
    criterion = nn.CrossEntropyLoss()
    best_model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_targets = []
    all_predicted = []

    with torch.no_grad():  # Disable gradient computation
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            outputs = best_model(inputs)
            loss = criterion(outputs, targets)

            # Update statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            # Collect all targets and predictions
            all_targets.extend(targets.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{test_loss / (batch_idx + 1):.4f}",
                "Acc": f"{test_correct / test_total:.4f}"
            })

    # Calculate test accuracy, loss, and F1-score
    test_accuracy = test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    test_f1 = f1_score(all_targets, all_predicted, average="weighted")

    # Calculate per-class accuracy
    conf_matrix = confusion_matrix(all_targets, all_predicted)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Calculate classification report (includes precision, recall, F1-score, and support)
    class_report = classification_report(all_targets, all_predicted, target_names=[f"Class {i}" for i in range(NUM_CLASSES)])

    # Print test summary
    print(f"Test Loss: {avg_test_loss:.4f}, "
          f"Test Acc: {test_accuracy:.4f},"
          f"Test F1: {test_f1:.4f}")

    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"Class {i}: {acc:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(class_report)
    
    normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    emotion_dict = {
        0: "Angry", 1: "Disgust", 2: "Fear",
        3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"
    }
    emotion_labels = [emotion_dict[i] for i in range(len(emotion_dict))]

    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

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

    best_model = ViTMobilenet(num_classes=len(class_names), 
                     in_channels=3,  
                     num_heads=12, 
                     embedding_dim=768, 
                     num_transformer_layers=12,
                     mlp_size=3072)
    best_model = best_model.to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    
    evaluate_model(best_model, test_loader)