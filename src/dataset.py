import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class GrainSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_filenames = []
        for img_fn in os.listdir(image_dir):
            if not img_fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                continue

            base_name = os.path.splitext(img_fn)[0]
            mask_fn_seg_png = f"{base_name}_seg.png"

            mask_fn_orig_ext = img_fn

            if os.path.exists(os.path.join(mask_dir, mask_fn_seg_png)):
                self.image_filenames.append((img_fn, mask_fn_seg_png))  # Pohrani par (ime_slike, ime_maske)
            elif os.path.exists(os.path.join(mask_dir, mask_fn_orig_ext)):
                self.image_filenames.append((img_fn, mask_fn_orig_ext))
            else:
                print(
                    f"⚠️ Upozorenje: Maska nije pronađena za '{img_fn}' (očekivano: '_seg.png' ili originalna ekstenzija). Preskačem.")

        if not self.image_filenames:
            raise ValueError(
                f"Nema uparenih slika i maski pronađenih u {image_dir} i {mask_dir}. Provjerite putanje i imena datoteka.")

        print(f"Dataset inicijalizovan sa {len(self.image_filenames)} uparenih slika/maski iz '{image_dir}'")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_fn, mask_fn = self.image_filenames[idx]

        img_path = os.path.join(self.image_dir, img_fn)
        image = Image.open(img_path).convert("RGB")

        mask_path = os.path.join(self.mask_dir, mask_fn)
        mask = Image.open(mask_path).convert("L")

        if self.image_transform and self.mask_transform:
            seed = random.randint(0, 2 ** 32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)

            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        elif self.image_transform:
            image = self.image_transform(image)
            mask = transforms.ToTensor()(mask)

        mask = (mask > 0.5).float()

        return image, mask, img_fn