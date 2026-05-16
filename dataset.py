import os
import glob
import random
import sqlite3
import json
import torch
import tarfile
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def setup_caption_db(db_path, jsonl_path):
    """
    Setup a correct and fast SQLite DB from the JSONL file if the 'type' column does not exist (runs only on first execution).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("PRAGMA table_info(captions)")
    cols = [r[1] for r in cur.fetchall()]
    
    if 'type' not in cols:
        print("=> Updating database structure (First setup only)... Setting up RAM-friendly structure.")
        cur.execute("DROP TABLE IF EXISTS captions")
        cur.execute("CREATE TABLE captions (file_name TEXT, type TEXT, text TEXT)")
        cur.execute("CREATE INDEX idx_filename ON captions(file_name)")
        
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found, database might be incomplete!")
        else:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                batch = []
                for line in f:
                    obj = json.loads(line)
                    batch.append((obj['file_name'], obj['type'], obj['text']))
                    if len(batch) >= 10000:
                        cur.executemany("INSERT INTO captions VALUES (?, ?, ?)", batch)
                        batch = []
                if batch:
                    cur.executemany("INSERT INTO captions VALUES (?, ?, ?)", batch)
        conn.commit()
        print("=> Database indexing complete. It will now run at lightning speed.")
    conn.close()

class T2IDataset(Dataset):
    def __init__(self, dataset_config, image_size=512, caption_dropout=0.1):
        """
        Reads images from Tar format and texts from DB format.
        caption_dropout: Provides Unconditional Training.
        """
        self.dataset_config = dataset_config
        self.image_size = image_size
        self.caption_dropout = caption_dropout
        
        self.jsonl_path = dataset_config["caption_jsonl"]
        self.db_path = self.jsonl_path.replace(".jsonl", ".db")
        self.tar_dir = dataset_config["tar_dir"]
        
        setup_caption_db(self.db_path, self.jsonl_path)
        
        self.conn = None
        
        self.image_index = []
        tar_files = glob.glob(os.path.join(self.tar_dir, "*.tar"))
        if not tar_files:
            print(f"WARNING: No Tar files found in {self.tar_dir}!")
            
        print(f"=> Indexing {len(tar_files)} Tar archives...")
        for tpath in tar_files:
            with tarfile.open(tpath, "r") as tar:
                for member in tar:
                    if member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        self.image_index.append((tpath, member.name, member.offset_data, member.size))
                        
        print(f"=> Total {len(self.image_index)} images indexed from Tar.")
        
        self.image_index.sort(key=lambda x: x[1])

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.image_index)
        
    def _get_connection(self):
        """Returns a safe SQLite connection for workers."""
        if self.conn is None:
            self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        return self.conn
            
    def __getitem__(self, idx):
        tpath, fname, offset, size = self.image_index[idx]
        
        with open(tpath, "rb") as f:
            f.seek(offset)
            image_bytes = f.read(size)
            
        try:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image)
        except Exception:
            return self.__getitem__(random.randint(0, len(self)-1))
            
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT text FROM captions WHERE file_name=? AND type IN ('long', 'medium', 'short') ORDER BY RANDOM() LIMIT 1", (fname,))
        row = cur.fetchone()
        
        caption = row[0] if row else ""
        
        if random.random() < self.caption_dropout:
             caption = ""
             
        return image_tensor, caption

class SimpleFolderDataset(Dataset):
    def __init__(self, dataset_configs, image_size=512, caption_dropout=0.0):
        """
        Images are read directly from folder(s) but TEXT data is loaded via .jsonl/SQLite infrastructure.
        Multiple dataset dictionaries (dataset_configs) can be passed as a list.
        """
        self.dataset_configs = dataset_configs if isinstance(dataset_configs, list) else [dataset_configs]
        self.image_size = image_size
        self.caption_dropout = caption_dropout
        
        self.conn = None
        self.image_paths = []
        self.db_paths = []
        
        for ds_config in self.dataset_configs:
            if "images_dir" not in ds_config:
                continue
                
            jsonl_path = ds_config["caption_jsonl"]
            db_path = jsonl_path.replace(".jsonl", ".db")
            
            setup_caption_db(db_path, jsonl_path)
            self.db_paths.append(db_path)
            
            search_dir = ds_config["images_dir"]
                
            folder_images = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG'):
                folder_images.extend(glob.glob(os.path.join(search_dir, '**', ext), recursive=True))
                
            self.image_paths.extend(folder_images)
            
        self.image_paths = list(set(self.image_paths))
        self.image_paths.sort()
            
        print(f"=> Dataset Found: {len(self.image_paths)} images across {len(self.dataset_configs)} configurations.")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _get_connections(self):
        if self.conn is None:
            self.conn = []
            for db_p in self.db_paths:
                self.conn.append(sqlite3.connect(f"file:{db_p}?mode=ro", uri=True))
        return self.conn
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
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

