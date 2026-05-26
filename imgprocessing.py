import cv2
import numpy as np
import os
import time

def saver(path, img):
    cv2.imwrite(path, img)

def my_halftone(img, pathDir):
    start = time.time()
    b = img[:, :, 0] * 0.114
    g = img[:, :, 1] * 0.587
    r = img[:, :, 2] * 0.299
    gray = (b + g + r).astype(np.uint8)
    end = time.time()
    
    saver(os.path.join(pathDir, "My_Halftone.jpg"), gray)
    return gray, end - start


def my_simple_conv(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape
    
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    pad_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    result = np.zeros_like(img, dtype=np.float64)
    
    for y in range(h):
        for x in range(w):
            
            window = pad_img[y : y + k_h, x : x + k_w]
            
            new_pixel_color = np.sum(window * kernel)
            
            result[y, x] = new_pixel_color
            
    return result


def my_color_conv(img_bgr, kernel):

    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1] 
    r = img_bgr[:, :, 2] 
    

    b_conv = my_simple_conv(b, kernel)
    g_conv = my_simple_conv(g, kernel)
    r_conv = my_simple_conv(r, kernel)
    
    result_color = np.dstack((b_conv, g_conv, r_conv))
    
    return result_color

def my_gauss(img, pathDir):
    start = time.time()
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0
    
    blurred = my_color_conv(img, kernel) 

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    end = time.time()
    
    saver(os.path.join(pathDir, "My_Gauss.jpg"), blurred)
    return blurred, end - start

def cv2_canny(img, pathDir):
    start = time.time()
    edges = cv2.Canny(img, 100, 200)
    end = time.time()
    saver(os.path.join(pathDir, "CV2_Canny.jpg"), edges)
    return edges, end - start

def my_sobel(img, pathDir):
    start = time.time()
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0],
                        [1, 2, 1]])
    
    gx = my_color_conv(img, kernel_x)
    gy = my_color_conv(img, kernel_y)
    
    g = np.sqrt(gx**2 + gy**2)
    g = np.clip(g, 0, 255).astype(np.uint8)
    end = time.time()
    
    saver(os.path.join(pathDir, "My_Sobel.jpg"), g)
    return g, end - start

def cv2_halftone(img, pathDir):
    start = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    end = time.time()
    saver(os.path.join(pathDir, "CV2_Halftone.jpg"), gray)
    return gray, end - start

def cv2_gauss(img, pathDir):
    start = time.time()
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    end = time.time()
    saver(os.path.join(pathDir, "CV2_Gauss.jpg"), blurred)
    return blurred, end - start


def my_gamma(img, pathDir):
    start = time.time()
    gamma = 0.5
    gamma_img = 255.0 * (img / 255.0)**gamma
    gamma_img = np.clip(gamma_img, 0, 255).astype(np.uint8)
    end = time.time()
    
    saver(os.path.join(pathDir, "My_Gamma.jpg"), gamma_img)
    return gamma_img, end - start

