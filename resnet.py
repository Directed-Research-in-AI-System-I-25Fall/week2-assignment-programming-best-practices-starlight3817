import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

# use MNIST to train 
def to_three_channels(x):
    return x.repeat(3, 1, 1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(to_three_channels),
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

model = resnet50(weights=ResNet50_Weights.DEFAULT)

model.fc = torch.nn.Linear(model.fc.in_features, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

accuracy = 100 * correct / total
print(accuracy)