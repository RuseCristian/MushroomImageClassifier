import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Set paths for static dataset
base_dir = "mushrooms"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Set parameters
img_height, img_width = 400, 400
batch_size = 32

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Define data transformations for preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets into PyTorch DataLoader objects
data_loaders = {
    'train': DataLoader(datasets.ImageFolder(train_dir, transform=data_transforms['train']), batch_size=batch_size, shuffle=True),
    'validation': DataLoader(datasets.ImageFolder(val_dir, transform=data_transforms['test']), batch_size=batch_size, shuffle=False),
    'test': DataLoader(datasets.ImageFolder(test_dir, transform=data_transforms['test']), batch_size=batch_size, shuffle=False)
}


num_classes = len(datasets.ImageFolder(train_dir).classes)

# Model 1: MobileNetV2
model1 = models.mobilenet_v2(pretrained=True)
model1.classifier[1] = nn.Linear(model1.last_channel, num_classes)
model1 = model1.to(device)

# Model 2: ResNet18
model2 = models.resnet18(pretrained=True)
model2.fc = nn.Linear(model2.fc.in_features, num_classes)
model2 = model2.to(device)

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

# Training function
def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs=40, patience=5, model_name="model.pth"):
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_acc, val_acc, train_losses, val_losses = [], [], [], []
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()  #
        running_loss, correct = 0.0, 0

        # Training loop
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc.append(correct / len(train_loader.dataset))
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        val_acc.append(correct / len(val_loader.dataset))
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc[-1]:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc[-1]:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_name)
            print(f"Model saved as {model_name}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Trigger early stopping
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    return train_acc, val_acc, train_losses, val_losses, best_epoch

# Train the models
train_acc1, val_acc1, train_loss1, val_loss1, best_epoch1 = train_model(model1, optimizer1, criterion, data_loaders['train'], data_loaders['validation'], model_name="mobilenetv2.pth")
train_acc2, val_acc2, train_loss2, val_loss2, best_epoch2 = train_model(model2, optimizer2, criterion, data_loaders['train'], data_loaders['validation'], model_name="resnet18.pth")

# MobileNetV2 Plots
plt.figure(figsize=(10, 6))
plt.plot(train_acc1, label='Train Accuracy (MobileNetV2)')
plt.plot(val_acc1, label='Validation Accuracy (MobileNetV2)')
plt.axvline(best_epoch1 - 1, color='red', linestyle='--', label='Early Stopping Trigger (MobileNetV2)')
plt.title('MobileNetV2 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("mobilenetv2_accuracy.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_loss1, label='Train Loss (MobileNetV2)')
plt.plot(val_loss1, label='Validation Loss (MobileNetV2)')
plt.axvline(best_epoch1 - 1, color='red', linestyle='--', label='Early Stopping Trigger (MobileNetV2)')
plt.title('MobileNetV2 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("mobilenetv2_loss.png")
plt.show()

# ResNet18 Plots
plt.figure(figsize=(10, 6))
plt.plot(train_acc2, label='Train Accuracy (ResNet18)')
plt.plot(val_acc2, label='Validation Accuracy (ResNet18)')
plt.axvline(best_epoch2 - 1, color='red', linestyle='--', label='Early Stopping Trigger (ResNet18)')
plt.title('ResNet18 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("resnet18_accuracy.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_loss2, label='Train Loss (ResNet18)')
plt.plot(val_loss2, label='Validation Loss (ResNet18)')
plt.axvline(best_epoch2 - 1, color='red', linestyle='--', label='Early Stopping Trigger (ResNet18)')
plt.title('ResNet18 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("resnet18_loss.png")
plt.show()


def evaluate_model(model, test_loader, model_name):
    model.load_state_dict(torch.load(model_name))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=datasets.ImageFolder(test_dir).classes))


evaluate_model(model1, data_loaders['test'], "mobilenetv2.pth")
evaluate_model(model2, data_loaders['test'], "resnet18.pth")
