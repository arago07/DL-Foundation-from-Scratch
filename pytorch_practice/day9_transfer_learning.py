import torch.nn as nn
import torchvision.models as models

# 1. Download RestNet18
resnet = models.resnet18(weights='DEFAULT')

# 2. Stop calculating gradient(-> increasing learning speed)
for param in resnet.parameters():
    param.requires_grad = False

# 3. Check the number of input features
num_ftrs = resnet.fc.in_features

# 4. Classifier(1000개 -> 2개 분류)
resnet.fc = nn.Linear(num_ftrs, 2)

# ----------------------------------------------
# 1. Data Transforms Pipeline
# ----------------------------------------------
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), # 좌우 반전 -> 데이터 양 증가
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Statistical number for ImageNet
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --------------------------------------------
# 2. Download
# --------------------------------------------
import os
import zipfile
import requests

# setting data saving path
data_dir = './data'
zip_path = os.path.join(data_dir, 'hymenotera_data.zip')
url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

# if the folder is not there, create it
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download & Extract
if not os.path.exists(os.path.join(data_dir, 'hymenoptera_data')):
    print("\nDownload Data...")
    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content) # not a text, byte type data

    print("Extracting...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("Completed!")
else:
    print("\nThe data already exists.")

# ----------------------------------------------------
# 3. Data Loading
# ----------------------------------------------------
import torch
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader

# ImageFolder - Reading images in the folder & Transforming data
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'hymenoptera_data', 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'hymenoptera_data', 'val'), data_transforms['val'])
}

# DataLoader - Gathering 4 images(batch_size=4) & Delivering to the model
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=4, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=4, shuffle=True)
}

# Saving the Number of Data and the Class Names(to calculate accuracy)
dataset_sizes = {x: len(image_datasets[x])for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"\nLoading Completed. Class names: {class_names}")
print(f"[The Number of Images] - Training: {dataset_sizes['train']} | Test: {dataset_sizes['val']}")

# -----------------------------------------------
# 4. Setting Device, Loss, Optimizer
# -----------------------------------------------

# MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
resnet = resnet.to(device)

# Setting a loss function
criterion = nn.CrossEntropyLoss() # classifying

# Training the tail part only, NOT the entire model
optimizer_ft = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)


# ----------------------------------------------
# 5. Training Loop
# ----------------------------------------------
import time
import copy

def train_model(model, criterion, optimizer, num_epochs=5):
    since = time.time() # recording starting time

    # An variable which records the best gradient
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f'-' * 10)
        
        # Each Epoch take turns training and validating
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Setting training mode
            else:
                model.eval() # Setting evaluationn mode

            running_loss = 0.0
            running_corrects = 0

            # DataLoader에서 1 batch(4 images)씩 추출
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # initializing grad

                # Allowing calculating grads(pre-backward) only when it's in train mode
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # choosing an output with the highest possibility
                    loss = criterion(outputs, labels)

                    # Backward & step if it is train mode
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Calculating
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Average Loss & Accuracy when an epoch ends
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase] # accuracy <- transfer to 'double' type

            print(f"Phase {phase}:  Loss {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Break a record if the accuracy is higher than before(in validating mode)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # deepcopy(): NO RELATION with the original variable
    
    time_elapsed = time.time() - since
    print(f"\nTraining Completed!")

    print('-' * 50)
    print(f"\nTime: {time_elapsed // 60:.0f} mins {time_elapsed % 60:.0f} secs")
    print(f"The Highest Accuracy: {best_acc:.4f}")

    # Transplant the best grads to the model
    model.load_state_dict(best_model_wts)
    return model, history

# --------------------------------------------
# 6. Running Train(5 epochs)
# --------------------------------------------
resnet, history  = train_model(resnet, criterion, optimizer_ft, num_epochs=5)


"""
=====================================================
Result
=====================================================
The data already exists.

Loading Completed. Class names: ['ants', 'bees']
[The Number of Images] - Training: 244 | Test: 153

Epoch 1/5
----------
Phase train:  Loss 0.5684 | Acc: 0.6967
Phase val:  Loss 0.4280 | Acc: 0.7843

Epoch 2/5
----------
Phase train:  Loss 0.5779 | Acc: 0.7623
Phase val:  Loss 0.3637 | Acc: 0.8431

Epoch 3/5
----------
Phase train:  Loss 0.3841 | Acc: 0.8443
Phase val:  Loss 0.2003 | Acc: 0.9346

Epoch 4/5
----------
Phase train:  Loss 0.4291 | Acc: 0.8279
Phase val:  Loss 0.2155 | Acc: 0.9477

Epoch 5/5
----------
Phase train:  Loss 0.6681 | Acc: 0.7459
Phase val:  Loss 0.2006 | Acc: 0.9477

Training Completed!

Time: 0 mins 15 secs
The Highest Accuracy: 0.9477

"""

# -----------------------------------------------
# Upgrade 1: Visualizing Learning Curve
# -----------------------------------------------
import matplotlib.pyplot as plt
print("\nDrawing Graph...")

plt.figure(figsize=(12, 5))

# 1st Graph: Loss Change
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='o', color='purple')
plt.plot(history['val_loss'], label='Val Loss', marker='o', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2nd Graph: Accuracy Change
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', marker='o', color='purple')
plt.plot(history['val_acc'], label='Val Acc', marker='o', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Graphh 간격 조정
plt.tight_layout()

save_dir = '/Users/arago/Desktop/DLFoundation/DL-Foundation-from-Scratch/images/pytorch_practice'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'day9_learning_curve.png')

# Saving Graphs
plt.savefig(save_path, dpi=300)

print('-' * 50)
print("\nSuccessfully Saved the Graphs!")
print(f"\nAddress: {save_path}")

plt.show()

# -----------------------------------------
# Upgrade 2: Visualizing Model Prediction
# -----------------------------------------

import numpy as np

def visualize_predictions(model, num_images=6):
    print("\nVisualizing Prediction Results...")

    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')

                # Success - blue, Failure - red
                color = 'blue' if preds[j] == labels[j] else 'red'
                ax.set_title(f"Pred: {class_names[preds[j]]} | True: {class_names[labels[j]]}", color=color)
                
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))

                # 원래 색상으로 복구(un-normalize)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                ax.imshow(img)

                if images_so_far == num_images:
                    plt.tight_layout()

                    save_path_pred = os.path.join(save_dir, 'day9_prediction.png')
                    plt.savefig(save_path_pred, dpi=300)
                    print("Prediction is saved successfully!")
                    print(f"Address: {save_path_pred}")

                    plt.show()
                    return
                
visualize_predictions(resnet, num_images=6)
