# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取图像
# def read_image(image_path):
#     return cv2.imread(image_path, 0)

# # 直方图均衡化
# def histogram_equalization(img):
#     return cv2.equalizeHist(img)

# # 对数增强
# def log_transform(img):
#     c = 255 / (np.log(1 + np.max(img)))
#     log_image = c * (np.log(1 + img))
#     return np.array(log_image, dtype=np.uint8)

# # 幂次增强（伽马校正）
# def gamma_transform(img, gamma=1.0):
#     c = 255.0 / np.max(img)
#     gamma_image = c * (img / 255.0) ** gamma
#     return np.array(gamma_image * 255, dtype=np.uint8)

# # 主函数
# def main():
#     # 读取图像
#     img = read_image('shiyan/shiyan1-1.bmp')

#     # 直方图均衡化
#     equalized_img = histogram_equalization(img)
#     cv2.imwrite('equalized_image.bmp', equalized_img)

# # 对数增强
# def log_transform(img):
#     c = 255 / (np.log(1 + np.max(img))) 
#     img_offset = img + 1
#     log_image = c * (np.log(img_offset))
#     log_image = np.clip(log_image, 0, 255)
#     return np.array(log_image, dtype=np.uint8)

#     # 幂次增强
#     gamma_img = gamma_transform(img, gamma=0.5)
#     cv2.imwrite('gamma_image.bmp', gamma_img)

#     # 图像显示和对比
#     fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#     # 原图
#     axs[0, 0].imshow(img, cmap='gray')
#     axs[0, 0].set_title('Original Image')
#     axs[0, 0].axis('off')
#     # 直方图均衡化
#     axs[0, 1].imshow(equalized_img, cmap='gray')
#     axs[0, 1].set_title('Histogram Equalization')
#     axs[0, 1].axis('off')
#     # 对数增强
#     axs[1, 0].imshow(log_img, cmap='gray')
#     axs[1, 0].set_title('Log Transformation')
#     axs[1, 0].axis('off')
#     # 幂次增强
#     axs[1, 1].imshow(gamma_img, cmap='gray')
#     axs[1, 1].set_title('Gamma Correction')
#     axs[1, 1].axis('off')
#     # 显示图像
#     plt.tight_layout()
#     plt.show()

# # 运行主函数
# if __name__ == '__main__':
#     main()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
def read_image(image_path):
    return cv2.imread(image_path, 0)

# 直方图均衡化
def histogram_equalization(img):
    return cv2.equalizeHist(img)

# 对数增强
def log_transform(img):
    c = 255 / (np.log(1 + np.max(img)))
    img_offset = img.astype(np.float32) + 1
    log_image = c * (np.log(img_offset))
    return np.array(log_image, dtype=np.uint8)

# 幂次增强（伽马校正）
def gamma_transform(img, gamma=1.0):
    c = 255.0 / np.max(img)
    gamma_image = c * (img / 255.0) ** gamma
    return np.array(gamma_image * 255, dtype=np.uint8)

# 绘制直方图
def plot_histogram(ax, img, title):
    hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
    ax.bar(bins[:-1], hist, width=1)
    ax.set_title(title)
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Frequency')

# 主函数
def main():
    img = read_image('shiyan\image\p1-7.tif')
    equalized_img = histogram_equalization(img)
    log_img = log_transform(img)
    gamma_img = gamma_transform(img, gamma=3.8)

    fig, axs = plt.subplots(2, 4, figsize=(9, 5))

    # 原图
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original Image')
    plot_histogram(axs[1, 0], img, 'Histogram (Original)')

    # 直方图均衡化
    axs[0, 1].imshow(equalized_img, cmap='gray')
    axs[0, 1].set_title('Histogram Equalization')
    plot_histogram(axs[1, 1], equalized_img, 'Histogram (Equalized)')

    # 对数增强
    axs[0, 2].imshow(log_img, cmap='gray')
    axs[0, 2].set_title('Log Transformation')
    plot_histogram(axs[1, 2], log_img, 'Histogram (Log)')

    # 伽马增强
    axs[0, 3].imshow(gamma_img, cmap='gray')
    axs[0, 3].set_title('Gamma Correction')
    plot_histogram(axs[1, 3], gamma_img, 'Histogram (Gamma)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


