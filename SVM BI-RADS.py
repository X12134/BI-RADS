# ========================
# 数据探索性分析
# ========================
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import random

# 忽略警告
warnings.filterwarnings("ignore")

# 读取数据
mat = scipy.io.loadmat('OASBUD.mat')
data = mat['data']

images = []
labels = []

for i in range(100):
    sample = data[0, i]

    # 图像处理
    rf_image = sample['rf2']
    if isinstance(rf_image, np.ndarray):
        rf_image = rf_image[0] if rf_image.size == 1 else rf_image

    # 标签处理
    raw_label = sample['birads']
    if isinstance(raw_label, np.ndarray):
        raw_label = raw_label[0]
    birads = int(str(raw_label)[0])

    images.append(rf_image)
    labels.append(birads)

# 将标签 [3,4,5] 转成 [0,1,2]
labels = [l - 3 for l in labels]

print("BI-RADS 标签类别：", sorted(set(labels)))


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
# 数据处理
# ========================
processed_images = []

for img in images:
    processed_img = envelope_detection_log_compression(img)
    processed_img = cv2.resize(processed_img.astype(np.float32), (224, 224))
    processed_images.append(processed_img)

processed_images = np.array(processed_images)
labels = np.array(labels)

# 展开成向量
X = processed_images.reshape((processed_images.shape[0], -1))
y = labels

print(f"特征形状：{X.shape}, 标签形状：{y.shape}")


# ========================
# 划分训练集和测试集
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ========================
# 计算类别权重
# ========================
# 计算类别权重
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# 将类别权重转换为字典形式
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


# ========================
# 训练SVM，增加类别权重
# ========================
# 建立SVM模型（带标准化，增加类别权重）
svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, class_weight=class_weight_dict)
)

# 训练
svm_model.fit(X_train, y_train)


# ========================
# 模型评估
# ========================
# 预测
y_pred = svm_model.predict(X_test)

# 准确率
train_acc = accuracy_score(y_train, svm_model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print(f"训练集准确率：{train_acc*100:.2f}%")
print(f"测试集准确率：{test_acc*100:.2f}%")

# 分类报告
print("\n分类报告：\n", classification_report(y_test, y_pred, target_names=['BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5'],
            yticklabels=['BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - SVM')
plt.tight_layout()
plt.show()


# ========================
# 随机预测一张图
# ========================
class_names = ['BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']

# 随机选择一张测试集中的图像
idx = random.randint(0, len(X_test) - 1)
image = X_test[idx].reshape(224, 224)  # 将图像展现为 224x224 大小
true_label = y_test[idx]

# 使用训练好的模型进行预测
pred_label = svm_model.predict([X_test[idx]])[0]

# 绘制图像
plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {class_names[pred_label]}\nTrue: {class_names[true_label]}")
plt.axis('off')
plt.show()
