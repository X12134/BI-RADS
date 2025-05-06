# ========================
# 数据探索性分析
# ========================
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random

# 载入数据
mat = scipy.io.loadmat('OASBUD.mat')
data = mat['data']

images = []
labels = []

for i in range(100):
    sample = data[0, i]
    rf_image = sample['rf2']
    if isinstance(rf_image, np.ndarray):
        rf_image = rf_image[0] if rf_image.size == 1 else rf_image
    raw_label = sample['birads']
    if isinstance(raw_label, np.ndarray):
        raw_label = raw_label[0]
    birads = int(str(raw_label)[0])  # 提取数字部分
    images.append(rf_image)
    labels.append(birads)

print("BI-RADS 标签类别：", sorted(set(labels)))

# 标签归一化 [3,4,5] -> [0,1,2]
labels = [l - 3 for l in labels]

# ========================
# 包络检测 + 对数压缩函数
# ========================
def envelope_detection_log_compression(image, log_offset=1):
    rectified_image = np.abs(image)
    envelope = cv2.GaussianBlur(rectified_image, (5, 5), 0)
    envelope = np.maximum(envelope, 1e-10)
    compressed_image = np.log(envelope + log_offset)
    return compressed_image

# ========================
# 自定义 Dataset 类
# ========================
class BIRADSDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = envelope_detection_log_compression(image)
        image = cv2.resize(image.astype(np.float32), (224, 224))
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ========================
# 定义分类模型
# ========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ========================
# 数据准备
# ========================
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

train_dataset = BIRADSDataset(X_train, y_train)
test_dataset = BIRADSDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ========================
# 设备配置
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# ========== 训练函数 ==========
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# ========== 验证函数 ==========
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# ========== 主训练流程 ==========
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}% "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

# ========== 结果可视化 ==========
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='red')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ========================
# 随机预测一张图
# ========================
class_names = ['BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']

model.eval()
idx = random.randint(0, len(test_dataset) - 1)
image, true_label = test_dataset[idx]

input_image = image.unsqueeze(0).to(device)

output = model(input_image)
pred_label = torch.argmax(output, dim=1).item()

plt.imshow(image.squeeze(0), cmap='gray')
plt.title(f"Predicted: {class_names[pred_label]}\nTrue: {class_names[true_label]}")
plt.axis('off')
plt.show()

# ========================
# 混淆矩阵 + 分类报告
# ========================
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for images_batch, labels_batch in test_loader:
        images_batch = images_batch.to(device)
        outputs = model(images_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels_batch.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("\n分类报告：")
print(classification_report(y_true, y_pred, target_names=class_names))
