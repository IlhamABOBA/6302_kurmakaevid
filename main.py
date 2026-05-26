import cv2
import matplotlib.pyplot as plt
import loader
import imgprocessing as ip


def main():
    print("Загружаем базу данных...")
    ids = loader.get_paintings_ids("MetObjects.csv")
    
    print("Ищем и скачиваем картину...")
    dir_path, img_path = loader.download_random_painting(ids)
    
    img_bgr = cv2.imread(img_path)
    
    h, w = img_bgr.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    print("\n--- РЕЗУЛЬТАТЫ ---")
    
    my_gray, t1 = ip.my_halftone(img_bgr, dir_path)
    cv2_gray, t2 = ip.cv2_halftone(img_bgr, dir_path)
    print(f"ЧБ: Моё = {t1:.4f} сек | OpenCV = {t2:.4f} сек")

    my_blur, t1 = ip.my_gauss(img_bgr, dir_path)
    cv2_blur, t2 = ip.cv2_gauss(img_bgr, dir_path)
    print(f"Гаусс: Моё = {t1:.4f} сек | OpenCV = {t2:.4f} сек")

    my_sobel, t1 = ip.my_sobel(img_bgr, dir_path)
    cv2_canny, t2 = ip.cv2_canny(cv2_blur, dir_path)
    print(f"Границы: Мой Собель = {t1:.4f} сек | OpenCV Canny = {t2:.4f} сек")

    my_gamma, t1 = ip.my_gamma(img_bgr, dir_path)
    print(f"Гамма-коррекция: {t1:.4f} сек")

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    
    sobel_dark = ip.my_halftone(my_sobel,dir_path)
    diff = cv2.absdiff( sobel_dark, cv2_canny)

    images =[img_bgr, my_gray, my_blur, my_sobel,
              diff, cv2_gray, cv2_blur, cv2_canny]
              
    titles =["Оригинал", "Моё ЧБ", "Мой Гаусс", "Мой Собель",
              "Моя Гамма (Доп)", "CV2 ЧБ", "CV2 Гаусс", "CV2 Canny"]

    for ax, img, title in zip(axes.flat, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        ax.set_title(title)
        ax.axis('off')

    plt.show()



if __name__ == '__main__':
    main()