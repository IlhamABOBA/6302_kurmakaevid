import numpy as np
import time
import os
import csv
import random
import cv2
import asyncio
import aiohttp
import aiofiles
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor


class Artwork(ABC):
    __slots__ = ('_name', '_pixels', '_metadata', 'index')

    def __init__(self, name: str, pixels: np.ndarray, metadata: dict, index: int) -> None:
        self._name = name
        self._pixels = pixels
        self._metadata = metadata
        self.index = index

    @property
    def name(self) -> str: return self._name
    @property
    def pixels(self) -> np.ndarray: return self._pixels
    @property
    def metadata(self) -> dict: return self._metadata

    @abstractmethod
    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        pass

    def blur_of_gauss(self) -> 'Artwork':
        kernel_gause = np.array([[1, 2, 1],[2, 4, 2], [1, 2, 1]]) / 16.0
        new_matrix = self.apply_filter(kernel_gause)
        new_matrix = np.clip(new_matrix, 0, 255).astype(np.uint8)
        return self.__class__(f"Gauss_{self.name}", new_matrix, self.metadata, self.index)

    def sobel_borders(self) -> 'Artwork':
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])

        gx = self.apply_filter(kernel_x)
        gy = self.apply_filter(kernel_y)

        g = np.sqrt(gx**2 + gy**2)
        g = np.clip(g, 0, 255).astype(np.uint8)
        return self.__class__(f"Sobel_{self.name}", g, self.metadata, self.index)


class ColorArtwork(Artwork):
    __slots__ = ('channels',)

    def __init__(self, name, pixels, metadata, index):
        super().__init__(name, pixels, metadata, index)
        self.channels = 3

    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        h, w, chanel = self.pixels.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        processed_layers =[]
        for i in range(chanel):
            pad_img = np.pad(self.pixels[:,:,i], ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            result = np.zeros_like(self.pixels[:,:,i], dtype=np.float64)

            for y in range(h):
                for x in range(w):
                    window = pad_img[y : y + k_h, x : x + k_w]
                    result[y, x] = np.sum(window * kernel)
            processed_layers.append(result)

        return np.dstack(processed_layers)

    def to_grayscale(self) -> 'BlackWhiteArtwork':
        b = self._pixels[:, :, 0] * 0.114
        g = self._pixels[:, :, 1] * 0.587
        r = self._pixels[:, :, 2] * 0.299
        gray = (b + g + r).astype(np.uint8)
        return BlackWhiteArtwork(self.name, gray, self.metadata, self.index)


class BlackWhiteArtwork(Artwork):
    __slots__ = ('channels',)

    def __init__(self, name, pixels, metadata, index):
        super().__init__(name, pixels, metadata, index)
        self.channels = 1

    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        h, w = self.pixels.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
         
        pad_img = np.pad(self.pixels, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(self.pixels, dtype=np.float64)

        for y in range(h):
            for x in range(w):
                window = pad_img[y : y + k_h, x : x + k_w]
                result[y, x] = np.sum(window * kernel)

        return result


def run_convolution_pipeline(artwork: Artwork) -> dict:
    pid = os.getpid()
    print(f"Convolution for image {artwork.index} started (PID {pid})")
    
    if isinstance(artwork, ColorArtwork):
        bw_art = artwork.to_grayscale()
    else:
        bw_art = artwork
        
    gauss_art = bw_art.blur_of_gauss()
    
    sobel_art = gauss_art.sobel_borders()
    
    print(f"Convolution for image {artwork.index} finished (PID {pid})")
    
    return {
        "Halftone": bw_art,
        "Gauss": gauss_art,
        "Sobel": sobel_art
    }

class ImageProcessor:
    def __init__(self):
        self.output_dir = "lab4_results"
        os.makedirs(self.output_dir, exist_ok=True)

    async def download_image_async(self, session: aiohttp.ClientSession, ids_list: list, index: int) -> Artwork:
        print(f"Downloading image {index} started (PID{os.getpid()})")
        
        while True:
            obj_id = random.choice(ids_list)
            url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
            
            async with session.get(url) as response:
                if response.status != 200: continue
                data = await response.json()
                
            img_url = data.get("primaryImageSmall")
            if not img_url: continue 

            async with session.get(img_url) as response:
                if response.status != 200: continue
                content = await response.read()

            orig_filename = f"{index}_{obj_id}_original.png"
            async with aiofiles.open(os.path.join(self.output_dir, orig_filename), mode="wb") as f:
                await f.write(content)

            nparray = np.frombuffer(content, np.uint8)
            pixels = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

            print(f"Downloading image {index} finished (PID{os.getpid()})")
            return ColorArtwork(data.get('title', 'Unknown'), pixels, data, index)

    async def save_artwork_async(self, artwork: Artwork, stage_name: str):
        
        obj_id = artwork.metadata.get('objectID')
        
        filename = f"{artwork.index}_{obj_id}_{stage_name}.png"
        
        is_success, buffer = cv2.imencode(".png", artwork.pixels)
        if is_success:
            async with aiofiles.open(os.path.join(self.output_dir, filename), mode="wb") as f:
                await f.write(buffer.tobytes())

    async def run(self, count: int):

        ids =[]

        with open('MetObjects.csv', mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            ids = [row["Object ID"] for row in reader if row["Classification"] == "Paintings"]
        
        async with aiohttp.ClientSession() as session:
            download_tasks =[self.download_image_async(session, ids, i) for i in range(1, count + 1)]
            artworks = await asyncio.gather(*download_tasks)

        print("\nЗапуск параллельной свёртки")
        with ProcessPoolExecutor() as executor:
            results_dicts = list(executor.map(run_convolution_pipeline, artworks))

        print("\nАсинхронное сохранение результатов")
        save_tasks =[]
        for stages_dict in results_dicts:
            for stage_name, art_obj in stages_dict.items():
                save_tasks.append(self.save_artwork_async(art_obj, stage_name))
        
        await asyncio.gather(*save_tasks)
        print("Сохранение завершено.")


if __name__ == "__main__":
    img_count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    
    total_start = time.perf_counter()
    
    processor = ImageProcessor()
    asyncio.run(processor.run(img_count))
    
    print(f"\nИтоговое время выполнения программы: {time.perf_counter() - total_start:.4f} сек.")