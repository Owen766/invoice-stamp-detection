import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_PATH = r"D:\\stamp_output\invoice.jpg"   # 图片路径
OUTPUT_DIR = r"D:\\stamp_output"   # 结果保存文件夹

# 检测单张图片
def detect_stamp(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：找不到图片 {image_path}")
        return 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
           cv2.inRange(hsv, lower_red2, upper_red2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    stamp_count = 0
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"Stamp {stamp_count+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 裁剪并保存印章区域
            crop = img[y:y+h, x:x+w]
            crop_path = os.path.join(output_dir, f"{base_name}_stamp_{stamp_count+1}.jpg")
            cv2.imwrite(crop_path, crop)

            stamp_count += 1

    print(f"[{os.path.basename(image_path)}] 检测到 {stamp_count} 个印章区域")
    return result, stamp_count


# ========== 执行 ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"错误：找不到图片 {IMAGE_PATH}，请检查路径。")
else:
    result, count = detect_stamp(IMAGE_PATH, OUTPUT_DIR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50]); upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Red Mask")
    plt.imshow(mask, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title(f"Detected ({count} stamps)")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "result.png"), dpi=150)
    plt.show()

    print(f"结果图已保存到：{OUTPUT_DIR}")
