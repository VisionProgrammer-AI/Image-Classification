import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch.nn as nn
import torch.optim as optim
import torch

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# for images, labels in train_loader:
#     print(f'image size is: {images.shape}')
#     print(f'Label number is: {labels[0]}')
#     # break
#     img = images[0].numpy()
#     img = np.transpose(img, (1,2,0))
#     cv2.imshow('MNIST Image', img)
#     cv2.waitKey(0)
#     break

model = nn.Sequential( nn.Flatten(),
                       nn.Linear(28*28,128),
                       nn.ReLU(),
                       nn.Linear(128,10)
                       )

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

print('Training complete')
torch.save(model.state_dict(), 'mnist_model.pth')
print('Model saved as mnist_model.pth')