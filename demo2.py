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

# 调和均值滤波
def harmonic_mean_filter(img, kernel_size):
    pad_size = kernel_size // 2
    padded_img = np.pad(img, pad_size, mode='edge')
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = kernel_size * kernel_size / np.sum(1.0 / (window + 1e-6))
    
    return output

# 逆调和均值滤波
def contra_harmonic_mean_filter(img, kernel_size, Q=1.5):
    pad_size = kernel_size // 2
    padded_img = np.pad(img, pad_size, mode='edge')
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(window ** (Q + 1)) / (np.sum(window ** Q) + 1e-6)
    
    return output

# 主函数
def main():
    img = read_image('image/p2-1.tif')
    if img is None:
        print("错误：无法读取图像文件。请检查文件路径是否正确。")
        return
    
    kernel_size = 3  # 滤波器大小

    # 计算各种滤波结果
    harmonic_img = harmonic_mean_filter(img, kernel_size)
    contra_harmonic_img = contra_harmonic_mean_filter(img, kernel_size)
    median_img = median_filter(img, kernel_size)

    # 创建2x2的子图
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # 原图
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('原始图像')
    axs[0, 0].axis('off')

    # 调和均值滤波
    axs[0, 1].imshow(harmonic_img, cmap='gray')
    axs[0, 1].set_title('调和均值滤波')
    axs[0, 1].axis('off')

    # 逆调和均值滤波
    axs[1, 0].imshow(contra_harmonic_img, cmap='gray')
    axs[1, 0].set_title('逆调和均值滤波 (Q=1.5)')
    axs[1, 0].axis('off')

    # 中值滤波
    axs[1, 1].imshow(median_img, cmap='gray')
    axs[1, 1].set_title('中值滤波')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
