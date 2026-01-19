# 简单的训练脚本模板
# 假设图片已保存在vcg_lamp_dataset，按类别分文件夹
# 这里只做分类训练的基本流程示例
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

DATA_DIR = 'vcg_lamp_dataset'  # 爬虫输出目录
MODEL_SAVE_PATH = 'models/lamp_recognition/lamp_model_resnet50.pth'  # 示例保存路径
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 2  # 需根据实际类别数调整

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.cuda() if torch.cuda.is_available() else model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.cuda(), labels.cuda() if torch.cuda.is_available() else (images, labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"模型已保存到: {MODEL_SAVE_PATH}")
