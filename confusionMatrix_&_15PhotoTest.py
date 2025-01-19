import os
import random
import torch
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from PIL import Image

base_dir = "mushrooms"
test_dir = os.path.join(base_dir, "test")

img_height, img_width = 400, 400
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
class_names = test_dataset.classes


def load_model(model_class, model_path, num_classes):
    model = model_class(pretrained=False)
    if isinstance(model, models.MobileNetV2):
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    elif isinstance(model, models.ResNet):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def predict_image(image, model):
    image_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    return predicted_idx.item(), confidence.item()


def display_confusion_matrix(model, model_name):
    print(f"Generating confusion matrix for {model_name}")
    y_true, y_pred = [], []

    for inputs, labels in torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical', cmap='viridis')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.show()


def visualize_predictions(model, model_name):
    print(f"Evaluating {model_name}")
    correct_predictions = 0

    indices = random.sample(range(len(test_dataset)), 15)
    selected_images = [test_dataset.samples[i] for i in indices]

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for idx, (image_path, label) in enumerate(selected_images):
        image = Image.open(image_path).convert('RGB')
        predicted_idx, confidence = predict_image(image, model)

        correct = (predicted_idx == label)
        if correct:
            correct_predictions += 1

        image_resized = image.resize((200, 200))
        axes[idx].imshow(image_resized)
        axes[idx].axis('off')
        axes[idx].set_title(
            f"Actual: {class_names[label]}\n"
            f"Predicted: {class_names[predicted_idx]}\n"
            f"Confidence: {confidence:.2f}",
            color="green" if correct else "red"
        )

    accuracy = correct_predictions / 15 * 100
    fig.suptitle(f"{model_name} - Correct Predictions: {correct_predictions}/15 ({accuracy:.2f}%)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"{model_name}_15_test_images.png")
    plt.show()


num_classes = len(class_names)

mobilenet_model = load_model(models.mobilenet_v2, "mobilenetv2.pth", num_classes)
resnet_model = load_model(models.resnet18, "resnet18.pth", num_classes)

display_confusion_matrix(mobilenet_model, "MobileNetV2")
display_confusion_matrix(resnet_model, "ResNet18")

visualize_predictions(mobilenet_model, "MobileNetV2")
visualize_predictions(resnet_model, "ResNet18")
