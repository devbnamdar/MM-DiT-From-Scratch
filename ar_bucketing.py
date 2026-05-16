import os
import glob
import json
import random
import tarfile
import sqlite3
from io import BytesIO
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

# 512px based Aspect Ratio (AR) Buckets
# Surface area is set to approximately ~262144 and edges are multiples of 64 for VAE/Patch.
BUCKETS_512 = [
    (512, 512), # 1:1 (Square)
    (576, 448), (448, 576), # ~4:3 / 3:4 (Traditional Photo)
    (640, 448), (448, 640), # ~1.42
    (640, 384), (384, 640), # ~5:3 / 3:5
    (704, 384), (384, 704), # ~16:9 / 9:16 (Wide Screen)
    (704, 320), (320, 704), # ~2.2:1 / 1:2.2 
    (768, 320), (320, 768), # ~21:9 / 9:21 (Ultra Wide)
    (768, 256), (256, 768), # ~3:1 / 1:3 (Panoramic)
    (832, 256), (256, 832), # ~3.25:1 (Extra Ultra Wide)
]

def get_closest_bucket(w, h, buckets):
    aspect_ratio = w / h
    best_bucket = None
    best_diff = float('inf')
    for bw, bh in buckets:
        bucket_ar = bw / bh
        diff = abs(aspect_ratio - bucket_ar)
        if diff < best_diff:
            best_diff = diff
            best_bucket = (bw, bh)
    return best_bucket

def create_ar_buckets(dataset_configs, output_name="ar_buckets.json"):
    if isinstance(dataset_configs, dict):
        dataset_configs = [dataset_configs]
        
    assignments = {}
    print(f"=> Reading image sizes and assigning buckets...")
    
    for ds_config in dataset_configs:
        if "tar_dir" in ds_config:
            tar_dir = ds_config["tar_dir"]
            tar_files = glob.glob(os.path.join(tar_dir, "*.tar"))
            if tar_files:
                print(f"   -> {len(tar_files)} TAR files are being processed: {tar_dir}")
                for tpath in tar_files:
                    with tarfile.open(tpath, "r") as tar:
                        for member in tar:
                            if member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                with open(tpath, "rb") as f:
                                    f.seek(member.offset_data)
                                    header = f.read(16000)
                                try:
                                    img = Image.open(BytesIO(header))
                                    w, h = img.size
                                    bw, bh = get_closest_bucket(w, h, BUCKETS_512)
                                    assignments[member.name] = {"w": w, "h": h, "bw": bw, "bh": bh}
                                except Exception:
                                    assignments[member.name] = {"w": 512, "h": 512, "bw": 512, "bh": 512}
                                    
        if "images_dir" in ds_config:
            search_dir = ds_config["images_dir"]
            image_paths = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG'):
                image_paths.extend(glob.glob(os.path.join(search_dir, '**', ext), recursive=True))
                
            image_paths = list(set(image_paths))
                
            if image_paths:
                print(f"   -> {len(image_paths)} images in folder are being processed: {search_dir}")
                for img_path in image_paths:
                    fname = os.path.basename(img_path)
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                        bw, bh = get_closest_bucket(w, h, BUCKETS_512)
                        assignments[fname] = {"w": w, "h": h, "bw": bw, "bh": bh}
                    except Exception:
                        assignments[fname] = {"w": 512, "h": 512, "bw": 512, "bh": 512}

    if os.path.isabs(output_name):
        output_path = output_name
    else:
        output_path = output_name
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"buckets": BUCKETS_512, "assignments": assignments}, f, indent=4, ensure_ascii=False)
    print(f"=> Aspect Ratio Buckets created successfully: {output_path}")

class ARBatchSampler(Sampler):
    """
    Sampler that puts images of the same size into the same batch for the Dataloader.
    """
    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buckets = dataset.buckets
        
    def __iter__(self):
        batches = []
        # Group image indices in each bucket to create batches
        for bucket_size, indices in self.buckets.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # Shuffle the order of batches themselves
        random.shuffle(batches)
        for batch in batches:
            yield batch
            
    def __len__(self):
        count = 0
        for indices in self.buckets.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count

class ARBucketDataset(Dataset):
    """
    Version using Aspect Ratio Bucketing. 
    Supports both TAR structure and standard Folder structures.
    """
    def __init__(self, dataset_configs, bucket_json="ar_buckets.json", caption_dropout=0.1):
        self.dataset_configs = dataset_configs if isinstance(dataset_configs, list) else [dataset_configs]
        self.caption_dropout = caption_dropout
        
        self.conn = None
        self.db_paths = []
        
        # Load Bucket JSON data (Searched in the first folder or provided directly)
        json_path = bucket_json
        if not os.path.exists(json_path):
             raise FileNotFoundError(f"{json_path} not found! Please run the ar_bucketing.py script first to create an index.")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assignments = data["assignments"]
        
        self.image_index = []
        
        for ds_config in self.dataset_configs:
            # DB Paths (For Caption readings)
            from dataset import setup_caption_db
            jsonl_path = ds_config["caption_jsonl"]
            db_path = jsonl_path.replace(".jsonl", ".db")
            setup_caption_db(db_path, jsonl_path)
            self.db_paths.append(db_path)
            
            if "tar_dir" in ds_config:
                tar_dir = ds_config["tar_dir"]
                tar_files = glob.glob(os.path.join(tar_dir, "*.tar"))
                for tpath in tar_files:
                    with tarfile.open(tpath, "r") as tar:
                        for member in tar:
                            if member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                self.image_index.append((True, tpath, member.name, member.offset_data, member.size))
                                
            if "images_dir" in ds_config:
                search_dir = ds_config["images_dir"]
                folder_images = []
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG'):
                    folder_images.extend(glob.glob(os.path.join(search_dir, '**', ext), recursive=True))
                    
                folder_images = list(set(folder_images))
                    
                for img_path in folder_images:
                    fname = os.path.basename(img_path)
                    self.image_index.append((False, img_path, fname, 0, 0))
                        
        self.image_index.sort(key=lambda x: x[2])
        
        self.buckets = defaultdict(list)
        self.index_to_bucket_size = {}
        
        for idx, item in enumerate(self.image_index):
            fname = item[2]
            if fname in assignments:
                bw = assignments[fname]["bw"]
                bh = assignments[fname]["bh"]
            else:
                bw, bh = 512, 512
            
            self.buckets[(bw, bh)].append(idx)
            self.index_to_bucket_size[idx] = (bw, bh)
            
    def __len__(self):
        return len(self.image_index)
        
    def _get_connections(self):
        if self.conn is None:
            self.conn = []
            for db_p in self.db_paths:
                self.conn.append(sqlite3.connect(f"file:{db_p}?mode=ro", uri=True))
        return self.conn
        
    def transform_image(self, image, bw, bh):
        # Aspect ratio might break during resize, so 
        # we first resize to fill at least bh/bw ratio and then crop.
        target_ar = bw / bh
        im_ar = image.width / image.height

        if im_ar > target_ar:
            # Image is wider than target area, so resize by height
            new_h = bh
            new_w = int(bh * im_ar)
        else:
            # Image is taller than target area, so resize by width
            new_w = bw
            new_h = int(bw / im_ar)

        transform = transforms.Compose([
            transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop((bh, bw)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(image)

    def __getitem__(self, idx):
        is_tar, source_path, fname, offset, size = self.image_index[idx]
        bw, bh = self.index_to_bucket_size[idx]
        
        try:
            if is_tar:
                with open(source_path, "rb") as f:
                    f.seek(offset)
                    image_bytes = f.read(size)
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            else:
                image = Image.open(source_path).convert('RGB')
                
            image_tensor = self.transform_image(image, bw, bh)
        except Exception:
            return self.__getitem__(random.randint(0, len(self)-1))
            
        conns = self._get_connections()
        caption = ""
        
        for conn in conns:
            cur = conn.cursor()
            cur.execute("SELECT text FROM captions WHERE file_name=? AND type IN ('long', 'medium', 'short') ORDER BY RANDOM() LIMIT 1", (fname,))
            row = cur.fetchone()
            if row:
                caption = row[0]
                break
        
        if random.random() < self.caption_dropout:
             caption = ""
             
        return image_tensor, caption

if __name__ == "__main__":
    print("Please use 'python run_ar_bucketing.py' to generate AR buckets from your config.py settings.")
