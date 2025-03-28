import timm
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet34, ResNet34_Weights

torch.manual_seed(12)
np.random.seed(12)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_dir = './data/'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')
saved_model_path = 'modified_resnet_model94.pth'

# variables
batch_size = 128
num_epochs = 10
num_classes = 100
learning_rate = 0.00001


class TestImageDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(
            test_dir) if os.path.isfile(os.path.join(test_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # For test data, we'll use the image name instead of a label
        return image, img_name


stronger_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(
        0.08, 1.0)),  # More variation in scale
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),  # Add vertical flips
    transforms.RandomRotation(30),  # More rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.RandomAffine(
        degrees=0, translate=(0.1, 0.1)),  # Add translation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

normal_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(root=train_dir,
                                  transform=normal_transforms['train']),
    'val': datasets.ImageFolder(root=val_dir,
                                transform=normal_transforms['val']),
    'test': TestImageDataset(test_dir,
                             normal_transforms['test'])
}
print("Class to idx mapping:", image_datasets['train'].class_to_idx)

# Create dataloaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size,
                        shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size,
                      shuffle=False, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size,
                       shuffle=False, num_workers=4)
}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
print(f"Number of training images: {dataset_sizes['train']}")
print(f"Number of validation images: {dataset_sizes['val']}")
print(f"Number of test images: {dataset_sizes['test']}")


class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=100, use_pretrained=True):
        super(ModifiedResNet, self).__init__()

        # Load pretrained ResNet34
        if use_pretrained:
            self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet34(weights=None)

        # Get the number of features in the final layer
        num_ftrs = self.model.fc.in_features

        # Add dropout before the final layer for regularization
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Initialize the model
# model = ModifiedResNet(use_pretrained=True)


class ModifiedTimMResNet(nn.Module):
    def __init__(self, num_classes=100, use_pretrained=True):
        super(ModifiedTimMResNet, self).__init__()

        # Choose one of the recommended models
        self.model = timm.create_model(
            'resnetrs101', pretrained=use_pretrained)

        # Get the number of features in the final layer
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 1024),  # Wider hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(1024),  # Add batch normalization
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Initialize the model
model = ModifiedTimMResNet(num_classes=num_classes, use_pretrained=True)

if os.path.exists(saved_model_path):
    print(f"Loading previously trained weights from {saved_model_path}")
    # Load weights
    model.load_state_dict(torch.load(saved_model_path))
    print("Model weights loaded successfully")
else:
    print("No previous model weights found. Starting training from scratch.")

model = model.to(device)

# Count the number of parameters


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_params = count_parameters(model)
print(f"Number of trainable parameters: {num_params:,}")
assert num_params < 100000000, "Model has too many parameters!"


def train_model(model, loss_func, optimizer, scheduler, epochs=25):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    best_weight = copy.deepcopy(model.state_dict()),
    best_acc = 0.0
    low_loss = 10000

    for ep in range(epochs):
        print("current epoch: ", ep)

        def train_detail(data, is_train, model,
                         loss_func, optimizer, scheduler):
            nonlocal best_acc
            nonlocal low_loss
            nonlocal best_weight
            nonlocal history

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Enable gradient tracking, backward and optimize
                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_func(outputs, labels)

                    if is_train:
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / \
                dataset_sizes['train' if is_train else 'val']
            epoch_acc = running_corrects.double(
            ) / dataset_sizes['train' if is_train else 'val']

            # update the history
            if is_train:
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                # Step the scheduler based on validation accuracy
                scheduler.step(epoch_acc)

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best model so far
            if not is_train and epoch_loss < low_loss:
                # best_acc= epoch_acc
                low_loss = epoch_loss
                best_weight = copy.deepcopy(model.state_dict())

                if ep > 10:
                    # Save the trained model
                    torch.save(model.state_dict(),
                               'modified_resnet_model_newest.pth')

        model.train()
        train_detail(dataloaders['train'], True, model,
                     loss_func, optimizer, scheduler)
        model.eval()
        train_detail(dataloaders['val'], False, model,
                     loss_func, optimizer, scheduler)

    # Load best model weights
    model.load_state_dict(best_weight)

    return model, history


# Optimizer and learning rate scheduler
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Learning rate scheduler that reduces LR when validation accuracy plateaus
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=3)

# optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
# lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)


model, history = train_model(
    model, loss_func, optimizer, scheduler, num_epochs)

# Save the trained model
torch.save(model.state_dict(), 'modified_resnet_model_newest.pth')

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()


def predict_test_data(model, dataloader):
    model.eval()

    # First, get the class to index mapping from your training dataset
    class_to_idx = image_datasets['train'].class_to_idx

    # Create the inverse mapping (from index to actual class)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # converts the model's prediction index to the actual class number
    index_remapping = {idx: int(cls_name)
                       for idx, cls_name in idx_to_class.items()}

    print("Index remapping:", index_remapping)

    image_names = []
    predictions = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, img_names in tqdm(dataloader,
                                      desc="Generating test predictions"):
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Apply the remapping to get actual class numbers
            remapped_preds = [index_remapping[p.item()] for p in preds]

            # Remove file extensions from image names
            clean_img_names = [os.path.splitext(
                img_name)[0] for img_name in img_names]

            # Store image names (without extensions) and remapped predictions
            image_names.extend(clean_img_names)
            predictions.extend(remapped_preds)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'image_name': image_names,
        'pred_label': predictions
    })

    # Check distribution of predictions after remapping
    print("\nPrediction distribution after remapping:")
    print(submission_df['pred_label'].value_counts().sort_index().head(20))

    # Check if any image names still have extensions
    if any('.' in name for name in submission_df['image_name']):
        print("WARNING: Some image names still contain '.' characters!")

    return submission_df


def evaluate_validation(model, dataloader):
    model.eval()

    running_corrects = 0
    all_preds = []
    all_labels = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader,
                                   desc="Evaluating validation set"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Statistics
            running_corrects += torch.sum(preds == labels.data)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = running_corrects.double() / len(dataloader.dataset)
    print(f'Validation Accuracy: {accuracy:.4f}')

    return all_preds, all_labels, accuracy


val_preds, val_labels, val_accuracy = evaluate_validation(
    model, dataloaders['val'])

# Generate predictions for test set with correct class mapping
prediction_df = predict_test_data(model, dataloaders['test'])
prediction_df.to_csv('prediction.csv', index=False)
print(f"submission file created with {len(prediction_df)} predictions.")
print(prediction_df.head())
