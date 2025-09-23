import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 123
IMG_SIZE = 224
BATCH_SIZE = 64

class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        # Ekstrak label dan piksel
        self.labels = self.dataframe['emotion'].values
        self.pixels = self.dataframe['pixels'].apply(self.string_to_image).values

    def string_to_image(self, pixels_string):
        # Konversi string piksel menjadi numpy array dan reshape ke 48x48
        pixels = np.array(pixels_string.split(), dtype='float32')
        image = pixels.reshape(48, 48)
        image = np.expand_dims(image, axis=-1)  # Tambahkan channel dimensi
        return image

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.pixels[idx]
        label = self.labels[idx]
        
        image = Image.fromarray(image.squeeze().astype('uint8'), mode='L')

        # Jika ada transformasi, terapkan ke image
        if self.transform:
            image = self.transform(image)

        return image, label
    
def create_transforms():
    # Create transform pipeline manually
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
        transforms.RandomRotation(10),     # Randomly rotate by 10 degrees
        transforms.RandomResizedCrop(
            size=IMG_SIZE,  # Output size
            scale=(0.8, 1.0)  # Range of the random crop size relative to the input size
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]) 

    # Create transform pipeline manually
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return train_transforms, test_transforms

def load_and_split_data(data_path):
    data = pd.read_csv(data_path)
    data_train, data_test = train_test_split(data, test_size=0.1, stratify=data['emotion'], random_state=SEED)
    data_train, data_val = train_test_split(data_train, test_size=0.1, stratify=data_train['emotion'], random_state=SEED)
    return data_train, data_val, data_test

def create_datasets(data_train, data_val, data_test, train_transforms, test_transforms):
    train_dataset = FERDataset(data_train, transform=train_transforms)
    val_dataset = FERDataset(data_val, transform=test_transforms)
    test_dataset = FERDataset(data_test, transform=test_transforms)
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            generator=torch.Generator().manual_seed(SEED))
    return train_loader, val_loader, test_loader