import os
import glob
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomPairedDataset(Dataset):
    """A dataset that supports selecting different subsets from the same folder."""
    
    def __init__(self, lq_folder, hq_folder, image_size=512, split='all', train_ratio=0.8, seed=42):
        """
        Args:
            lq_folder: Folder containing low-quality images.
            hq_folder: Folder containing high-quality images.
            image_size: Image size.
            split: One of 'train', 'val', or 'all'.
            train_ratio: Proportion of images used for training.
            seed: Random seed used to ensure a consistent split.
        """
        super().__init__()
        self.lq_folder = lq_folder
        self.hq_folder = hq_folder
        self.image_size = image_size
        
        # Get all filenames.
        self.lq_files = sorted(os.listdir(lq_folder))
        
        # Set the random seed to ensure a consistent split.
        random.seed(seed)
        
        # Shuffle the file list.
        all_indices = list(range(len(self.lq_files)))
        random.shuffle(all_indices)
        
        # Select a subset based on the split type.
        if split == 'train':
            split_idx = int(len(all_indices) * train_ratio)
            self.indices = all_indices[:split_idx]
        elif split == 'val':
            split_idx = int(len(all_indices) * train_ratio)
            self.indices = all_indices[split_idx:]
        else:
            self.indices = all_indices
            # Add image transformations.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Dataset {split}: {len(self.indices)} images")


    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # Use the mapped index.
        file_idx = self.indices[idx]
        file_name = self.lq_files[file_idx]
        
        # Use the original loading logic below.
        lq_path = os.path.join(self.lq_folder, file_name)
        hq_path = os.path.join(self.hq_folder, file_name)
        
        # Load images.
        lq_img = Image.open(lq_path).convert('RGB')
        hq_img = Image.open(hq_path).convert('RGB')
        
        # Apply transformations.
        lq_tensor = self.transform(lq_img)
        hq_tensor = self.transform(hq_img)
        
        
        # Return a dictionary.
        return {
            'lq': lq_tensor,
            'gt': hq_tensor,
            'filename': os.path.basename(lq_path)
        }

class CelebARandomPairedDataset(Dataset):
    """Pair CelebA test and validation images in filename order."""
    
    def __init__(self, lq_folder, hq_folder, image_size=512, limit=None):
        """
        Args:
            lq_folder: Path to the low-quality image folder.
            hq_folder: Path to the high-quality image folder.
            image_size: Image size.
            limit: Optional maximum number of images.
        """
        super().__init__()
        self.lq_folder = lq_folder
        self.hq_folder = hq_folder
        self.image_size = image_size
        
        # Get sorted file lists.
        self.lq_files = sorted(os.listdir(lq_folder))
        self.hq_files = sorted(os.listdir(hq_folder))
        
        # Ensure that the file counts match, or handle a mismatch.
        min_length = min(len(self.lq_files), len(self.hq_files))
        if min_length != len(self.lq_files) or min_length != len(self.hq_files):
            print(f"Warning: The numbers of LQ files ({len(self.lq_files)}) and HQ files ({len(self.hq_files)}) do not match")
            self.lq_files = self.lq_files[:min_length]
            self.hq_files = self.hq_files[:min_length]
        
        # Limit the number of files if requested.
        if limit is not None and limit > 0 and limit < min_length:
            self.lq_files = self.lq_files[:limit]
            self.hq_files = self.hq_files[:limit]
        
        # Store image-pair information.
        self.file_pairs = list(zip(self.lq_files, self.hq_files))
        
        # Image transformations.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.lq_files)
        
    def __getitem__(self, idx):
        # Get the corresponding LQ and HQ filenames.
        lq_file = self.lq_files[idx]
        hq_file = self.hq_files[idx]
        
        # Build full paths.
        lq_path = os.path.join(self.lq_folder, lq_file)
        hq_path = os.path.join(self.hq_folder, hq_file)
        
        # Load images.
        try:
            lq_img = Image.open(lq_path).convert('RGB')
            hq_img = Image.open(hq_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}, LQ: {lq_path}, HQ: {hq_path}")
            # Fall back to the next image pair.
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply transformations.
        lq_tensor = self.transform(lq_img)
        hq_tensor = self.transform(hq_img)
        
        # Return a dictionary that includes the filename.
        return {
            'lq': lq_tensor,
            'gt': hq_tensor,
            'filename': os.path.basename(lq_file)  # Use the LQ filename as the identifier.
        }
class RealImageDataset(Dataset):
    """A general real-image dataset loader without GT, suitable for any real-world image set."""
    def __init__(self, lq_folder, image_size=512, recursive=True, limit=None, extensions=('.png', '.jpg', '.jpeg', '.webp')):
        """
        Args:
            image_folder: Path to the image folder.
            image_size: Resized image dimensions.
            recursive: Whether to search subfolders recursively.
            limit: Optional maximum number of images to load.
            extensions: Supported image file extensions.
        """
        self.image_folder = lq_folder
        self.image_size = image_size
        
        # Get all image files.
        self.image_paths = []
        
        if recursive:
            # Search all subfolders recursively.
            for root, _, files in os.walk(lq_folder):
                for file in files:
                    if file.lower().endswith(extensions):
                        self.image_paths.append(os.path.join(root, file))
        else:
            # Search only the specified folder.
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(lq_folder, f"*{ext}")))
                self.image_paths.extend(glob.glob(os.path.join(lq_folder, f"*{ext.upper()}")))
        
        # Sort to ensure a consistent order.
        self.image_paths = sorted(self.image_paths)
        
        # Limit the number of images.
        if limit is not None and limit > 0 and limit < len(self.image_paths):
            self.image_paths = self.image_paths[:limit]
        
        # Default transformations.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Loaded {len(self.image_paths)} images from {lq_folder}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            img_tensor = self.transform(img)
            
            return {
                "lq": img_tensor,      
                "filename": os.path.basename(img_path)
            }
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((index + 1) % len(self))
