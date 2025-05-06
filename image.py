import scipy.io
import numpy as np
import cv2
import os

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
# 伪彩色合成函数
# ========================
def create_pseudo_color_image(base_image, mask_image, threshold=30):
    # 处理 base 图像
    base_processed = envelope_detection_log_compression(base_image)
    base_norm = cv2.normalize(base_processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    base_color = cv2.cvtColor(base_norm, cv2.COLOR_GRAY2BGR)

    # 处理 mask 图像
    mask_processed = envelope_detection_log_compression(mask_image)
    mask_norm = cv2.normalize(mask_processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_mask = cv2.threshold(mask_norm, threshold, 255, cv2.THRESH_BINARY)

    # 创建红色遮罩
    red_overlay = np.zeros_like(base_color)
    red_overlay[:, :, 2] = binary_mask  # 红色通道高亮

    # 合成伪彩色图像
    pseudo_color = cv2.addWeighted(base_color, 1.0, red_overlay, 0.6, 0)
    return pseudo_color

# ========================
# 主程序开始
# ========================
# 载入数据
mat = scipy.io.loadmat('OASBUD.mat')
data = mat['data']

# 遍历数据样本
for i in range(data.shape[1]):
    sample = data[0, i]

    # 提取 BI-RADS 标签
    raw_label = sample['birads']
    if isinstance(raw_label, np.ndarray):
        raw_label = raw_label[0]
    birads = int(str(raw_label)[0])  # 获取 BI-RADS 等级

    # 创建输出子目录
    birads_dir = os.path.join('OASBUD', f'birads{birads}')
    os.makedirs(birads_dir, exist_ok=True)

    # 处理原始图像并保存
    for key in ['rf1', 'rf2', 'roi1', 'roi2']:
        img = sample[key]
        if isinstance(img, np.ndarray):
            img = img[0] if img.size == 1 else img

        # 包络检测 + 对数压缩
        processed_img = envelope_detection_log_compression(img)

        # 归一化为 0~255 并转换为 uint8 类型保存为图像
        norm_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.uint8)

        # 保存图像
        filename = f'sample{i}_{key}.png'
        filepath = os.path.join(birads_dir, filename)
        cv2.imwrite(filepath, norm_img)

    # 合成并保存伪彩色图像（rf1+roi1 和 rf2+roi2）
    for pair in [('rf1', 'roi1'), ('rf2', 'roi2')]:
        base_img = sample[pair[0]]
        mask_img = sample[pair[1]]
        pseudo_color_img = create_pseudo_color_image(base_img, mask_img)

        # 保存伪彩色图像
        pseudo_filename = f'sample{i}_pseudo_{pair[0]}_{pair[1]}.png'
        pseudo_filepath = os.path.join(birads_dir, pseudo_filename)
        cv2.imwrite(pseudo_filepath, pseudo_color_img)

print("图像与伪彩色图像导出完成。")
