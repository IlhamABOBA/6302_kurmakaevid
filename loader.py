import csv
import requests
import os
import json
import random


def get_paintings_ids(filename):
    ids =[]
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Classification"] == "Paintings":
                ids.append(row["Object ID"])
    return ids

def download_random_painting(ids):
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
        
        dir_path = os.path.join("paintings", f"object_{rand_id}")
        os.makedirs(dir_path, exist_ok=True)
        
        json_path = os.path.join(dir_path, "info.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        jpg_path = os.path.join(dir_path, "img.jpg")
        with open(jpg_path, "wb") as f:
            f.write(img_response.content)
            
        return dir_path, jpg_path