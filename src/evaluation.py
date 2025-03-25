import timm
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights

torch.manual_seed(12)
np.random.seed(12)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_dir = './data/'
test_dir = os.path.join(data_dir, 'test')
saved_model_path = 'modified_resnet_model_best_94.pth'

# variables
batch_size = 128
num_epochs = 10
num_classes = 100
learning_rate = 0.00001
print(num_epochs, learning_rate)


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


normal_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'test': TestImageDataset(test_dir, normal_transforms['test'])
}

# Create dataloaders
dataloaders = {
    'test': DataLoader(image_datasets['test'], batch_size=batch_size,
                       shuffle=False, num_workers=4)
}


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


# Optimizer and learning rate scheduler
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Learning rate scheduler that reduces LR when validation accuracy plateaus
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=3)


# Function to create a fixed class_to_idx mapping
def create_class_mapping():
    # Sort class names (0-99) alphabetically, not numerically
    class_names = sorted([str(i) for i in range(100)])

    # Create mapping as class_name -> index
    class2idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    return class2idx


def predict_test_data(model, dataloader):
    model.eval()

    # Create fixed class mapping (0-99)
    class_to_idx = create_class_mapping()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    index_remapping = {idx: int(cls_name)
                       for idx, cls_name in idx_to_class.items()}

    print("Using direct class mapping (class 0->0, 1->1, etc.)")

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

    return submission_df


# Generate predictions for test set with correct class mapping
prediction_df = predict_test_data(model, dataloaders['test'])
prediction_df.to_csv('prediction.csv', index=False)
print(f"submission file created with {len(prediction_df)} predictions.")
print(prediction_df.head())
