import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 算术均值滤波
def arithmetic_mean_filter(img, kernel_size):
    pad_size = kernel_size // 2
    padded_img = np.pad(img, pad_size, mode='edge')
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.mean(padded_img[i:i + kernel_size, j:j + kernel_size])
    
    return output

# 几何均值滤波
def geometric_mean_filter(img, kernel_size):
    pad_size = kernel_size // 2
    padded_img = np.pad(img, pad_size, mode='edge')
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.exp(np.mean(np.log(padded_img[i:i + kernel_size, j:j + kernel_size] + 1e-6)))
    
    return output

# 中值滤波
def median_filter(img, kernel_size):
    pad_size = kernel_size // 2
    padded_img = np.pad(img, pad_size, mode='edge')
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.median(padded_img[i:i + kernel_size, j:j + kernel_size])
    
    return output

# 主函数
def main():
    img = read_image('image\p2-1.tif')
    kernel_size = 3  # 滤波器大小

    arith_mean_img = arithmetic_mean_filter(img, kernel_size)
    geom_mean_img = geometric_mean_filter(img, kernel_size)
    median_img = median_filter(img, kernel_size)

    fig, axs = plt.subplots(2, 2, figsize=(6, 6))

    # 原图
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # 算术均值滤波
    axs[0, 1].imshow(arith_mean_img, cmap='gray')
    axs[0, 1].set_title('Arithmetic Mean Filter')
    axs[0, 1].axis('off')

    # 几何均值滤波
    axs[1, 0].imshow(geom_mean_img, cmap='gray')
    axs[1, 0].set_title('Geometric Mean Filter')
    axs[1, 0].axis('off')

    # 中值滤波
    axs[1, 1].imshow(median_img, cmap='gray')
    axs[1, 1].set_title('Median Filter')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
