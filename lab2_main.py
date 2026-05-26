import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from functools import wraps
import csv
import os
import json
import requests
import random
import cv2

def log_decorator(func):

    @wraps(func)
    def wrapper(*args,**kwargs):
        print(f"\n[log {time.perf_counter()}] Начала работать {func.__name__}...")
        result = func(*args,**kwargs)
        print(f"[log {time.perf_counter()}] Закончила работать {func.__name__}...\n")
        return result
    
    return wrapper


def time_decorator(func):

    @wraps(func)
    def wrapper(*args, ** kwargs):
        start = time.perf_counter()
        result = func(*args,**kwargs)
        print(f"Функция {func.__name__} отработала за {time.perf_counter() - start}")

        return result

    return wrapper



class Artwork(ABC):
    
    def __init__(self, name: str, pixels: np.ndarray, metadata: dict) -> None:
        self._name = name
        self._pixels = pixels
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def pixels(self) -> np.ndarray:
        return self._pixels
    
    @property
    def metadata(self) -> dict:
        return self._metadata
    
    def __str__(self) -> str:
        return f"Картина: {self._name} | Разрешение: {self._pixels.shape} | Метаданные: {self._metadata}\n"

    def __add__(self, other: 'Artwork') -> 'Artwork':
        new_pixels = np.clip(self.pixels + other.pixels,0,255).astype(np.uint8)
        return self.__class__(f"{self._name} + {other._name}", new_pixels, {**self._metadata, **other._metadata} )


    @abstractmethod
    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        pass

    def blur_of_gauss(self) -> 'Artwork':
        print(f"Применяю Гаусса к {self.name}")
        kernel_gause = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]]) / 16.0

        new_matrix = self.apply_filter(kernel_gause)
        new_matrix = np.clip(new_matrix, 0, 255).astype(np.uint8)

        return self.__class__(f"Размытый {self.name}", new_matrix, self.metadata)

    def sobel_borders(self) -> 'Artwork':

        print(f"Применяю Собеля к {self.name}")

        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2], 
                             [-1, 0, 1]])
    
        kernel_y = np.array([[-1, -2, -1], 
                             [ 0,  0,  0],
                             [ 1,  2,  1]])

        gx = self.apply_filter(kernel_x)
        gy = self.apply_filter(kernel_y)

        g = np.sqrt(gx**2 + gy**2)
        g = np.clip(g, 0, 255).astype(np.uint8)

        return self.__class__(f"Собель {self.name}",g, self.metadata)
    
    def gamma(self) -> 'Artwork':

        print(f"Применяю гамму к {self.name}")
        gamma = 0.5
        gamma_pixels = 255.0 * (self.pixels / 255.0)**gamma
        gamma_pixels = np.clip(gamma_pixels, 0, 255).astype(np.uint8)

        return self.__class__(f"Гамма {self.name}",gamma_pixels, self.metadata)



class ColorArtwork(Artwork):

    __slots__ = ('_name','_pixels','_metadata','channels')

    def __init__(self, name, pixels, metadata):
        super().__init__(name, pixels, metadata)
        self.channels = 3

    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        
        h,w,chanel = self.pixels.shape
        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2

        processed_layers = []

        for i in range(chanel):
            
            pad_img = np.pad(self.pixels[:,:,i], ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

            result = np.zeros_like(self.pixels[:,:,i], dtype=np.float64)

            for y in range(h):
                for x in range(w):

                    window = pad_img[y : y + k_h, x : x + k_w]
                    new_pixel_color = np.sum(window * kernel)
                    result[y,x] = new_pixel_color
            
            processed_layers.append(result)

        return np.dstack(processed_layers)


    def to_grayscale(self) -> 'BlackWhiteArtwork':
        b = self._pixels[:, :, 0] * 0.114
        g = self._pixels[:, :, 1] * 0.587
        r = self._pixels[:, :, 2] * 0.299
        gray = (b + g + r).astype(np.uint8)

        return BlackWhiteArtwork(self.name,gray,self.metadata)

    
class BlackWhiteArtwork(Artwork):

    __slots__ = ('_name','_pixels','_metadata','channels')

    def __init__(self, name, pixels, metadata):
        super().__init__(name, pixels, metadata)
        self.channels = 1

    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:

        h,w = self.pixels.shape
        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2
         
        pad_img = np.pad(self.pixels, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        result = np.zeros_like(self.pixels, dtype=np.float64)

        for y in range(h):
            for x in range(w):

                window = pad_img[y : y + k_h, x : x + k_w]
                new_pixel_color = np.sum(window * kernel)
                result[y,x] = new_pixel_color

        return result


class ImageProcessor:

    @time_decorator
    @log_decorator
    def process_artwork(self, artwork: 'Artwork', dir_name: str) -> None:

        b_and_w = artwork.to_grayscale()
        cv2.imwrite(os.path.join(f"{dir_name}/object_{artwork.metadata.get('objectID')}", "My_Halftone.jpg"), b_and_w.pixels)
        blur_art = artwork.blur_of_gauss()
        cv2.imwrite(os.path.join(f"{dir_name}/object_{artwork.metadata.get('objectID')}", "My_Gauss.jpg"), blur_art.pixels)
        sobel_art = b_and_w.sobel_borders()
        cv2.imwrite(os.path.join(f"{dir_name}/object_{artwork.metadata.get('objectID')}", "My_Sobel.jpg"), sobel_art.pixels)
        Canny_art = cv2.Canny(artwork.pixels, 100, 200)
        cv2.imwrite(os.path.join(f"{dir_name}/object_{artwork.metadata.get('objectID')}", "CV2_Canny.jpg"), Canny_art)
        gray = cv2.cvtColor(artwork.pixels, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(f"{dir_name}/object_{artwork.metadata.get('objectID')}", "CV2_Halftone.jpg"), gray)
        blurred = cv2.GaussianBlur(artwork.pixels, (3, 3), 0)
        cv2.imwrite(os.path.join(f"{dir_name}/object_{artwork.metadata.get('objectID')}", "CV2_Gauss.jpg"), blurred)
        gamma = artwork.gamma()
        cv2.imwrite(os.path.join(f"{dir_name}/object_{artwork.metadata.get('objectID')}", "My_gamma.jpg"), gamma.pixels)

    @time_decorator
    @log_decorator
    def download_random_image(self, cvs_filename: str, dir_name_create: str) -> 'Artwork':
        ids =[]
        with open(cvs_filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["Classification"] == "Paintings":
                    ids.append(row["Object ID"])
            img_url = ""

        while not img_url:
            rand_id = random.choice(ids)
            url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{rand_id}"
            
            response = requests.get(url)
            data = response.json()
            
            img_url = data.get("primaryImageSmall")
            if not img_url:
                continue
                
            img_response = requests.get(img_url)
            
            dir_path = os.path.join(dir_name_create, f"object_{rand_id}")
            os.makedirs(dir_path, exist_ok=True)
            
            json_path = os.path.join(dir_path, "info.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            jpg_path = os.path.join(dir_path, "img.jpg")
            with open(jpg_path, "wb") as f:
                f.write(img_response.content)

        return ColorArtwork(data.get('title'),cv2.imread(jpg_path),data)

            
        
proc = ImageProcessor()

my_art = proc.download_random_image("MetObjects.csv","paintings")

print(my_art)


proc.process_artwork(my_art,"paintings")


bright_art = my_art + my_art 

cv2.imwrite(os.path.join(f"paintings/object_{bright_art.metadata.get('objectID')}", "Added.jpg"), bright_art.pixels)
