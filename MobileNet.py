import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from collections import Counter  # ✅ 新增

# 设置图像参数
img_size = (224, 224)
batch_size = 32

# 加载训练集和验证集
train_ds = tf.keras.utils.image_dataset_from_directory(
    'train6',
    labels='inferred',
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'val6',
    labels='inferred',
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# 获取类名
class_names = train_ds.class_names  # ['0', '1', '2']

# 显示部分训练集图像及其标签
def show_dataset_samples(dataset, class_names, title='Dataset Samples'):
    plt.figure(figsize=(10, 8))
    for images, labels in dataset.take(1):  # 只取一个 batch 展示
        for i in range(9):  # 显示前9张图像
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            label_index = np.argmax(labels[i].numpy())
            plt.title(f"Label: {class_names[label_index]}")
            plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 显示训练集和验证集样本
show_dataset_samples(train_ds, class_names, title='Train Dataset Samples')
show_dataset_samples(val_ds, class_names, title='Validation Dataset Samples')

# 性能优化
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ✅ 统计训练集中每个类别的数量
all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(np.argmax(labels.numpy()))
label_counts = Counter(all_labels)

# ✅ 计算类别权重（样本数少的类别权重大）
total = sum(label_counts.values())
class_weight = {
    i: total / (len(label_counts) * count)
    for i, count in label_counts.items()
}
print("Class weights:", class_weight)

# 构建模型
base_model = applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # 冻结预训练部分

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')  # 三分类
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 模型训练，✅ 使用类别权重
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weight
)

# 绘制训练过程图像
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 获取真实标签和预测标签
y_val_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in val_ds])
y_val_pred_prob = model.predict(val_ds)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)

# 混淆矩阵和分类报告
cm = confusion_matrix(y_val_true, y_val_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 打印分类报告
print("Classification Report:\n", classification_report(
    y_val_true,
    y_val_pred,
    target_names=[f'BI-RADS {int(c)+3}' for c in class_names]
))
