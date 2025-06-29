import os
import shutil
import pandas as pd
import random

ORIGINAL_IMAGES_FOLDER = "Images"
BINARY_MASKS_FOLDER = "Binary Masks"
GRAIN_COUNT_CSV = "grain_counts.csv"

# --- Putanje za organizaciju podataka (ovo će se generirati) ---
DATASET_SPLIT_BASE = "dataset_split_for_dinov2"
TRAIN_IMAGES_DIR = os.path.join(DATASET_SPLIT_BASE, "train", "images")
TRAIN_MASKS_DIR = os.path.join(DATASET_SPLIT_BASE, "train", "masks")
VAL_IMAGES_DIR = os.path.join(DATASET_SPLIT_BASE, "val", "images")
VAL_MASKS_DIR = os.path.join(DATASET_SPLIT_BASE, "val", "masks")
TEST_IMAGES_DIR = os.path.join(DATASET_SPLIT_BASE, "test", "images")
TEST_MASKS_DIR = os.path.join(DATASET_SPLIT_BASE, "test", "masks")

# --- Postavke za podjelu podataka ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# --- Kreiranje potrebne strukture foldera i kopiranje podataka ---
def setup_data_folders():
    # Očisti stare foldere ako postoje
    if os.path.exists(DATASET_SPLIT_BASE):
        shutil.rmtree(DATASET_SPLIT_BASE)

    for folder in [TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
                   VAL_IMAGES_DIR, VAL_MASKS_DIR,
                   TEST_IMAGES_DIR, TEST_MASKS_DIR]:
        os.makedirs(folder, exist_ok=True)
    print(f"Struktura foldera kreirana u '{DATASET_SPLIT_BASE}'")

    # Učitavanje CSV-a za brojanje zrna (ground truth)
    df_grain_counts = pd.read_csv(GRAIN_COUNT_CSV)
    # Stvori dictionary za brzi lookup: {ime_slike: broj_zrna}
    grain_counts_dict = {row['image_name']: row['grain_count'] for index, row in df_grain_counts.iterrows()}

    # Dohvati sve originalne slike i provjeri imaju li odgovarajuću masku
    all_image_filenames = [f for f in os.listdir(ORIGINAL_IMAGES_FOLDER) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]

    paired_files = []
    for img_fn in all_image_filenames:
        base_name = os.path.splitext(img_fn)[0]
        mask_fn_potential_png = f"{base_name}_seg.png"

        # Provjeri postoji li maska s originalnom ekstenzijom ili s .png ekstenzijom
        mask_path_orig_ext = os.path.join(BINARY_MASKS_FOLDER, img_fn)  # Maska s istom ekstenzijom kao original
        mask_path_png_ext = os.path.join(BINARY_MASKS_FOLDER, mask_fn_potential_png)  # Maska s .png ekstenzijom

        if os.path.exists(mask_path_orig_ext):
            paired_files.append((img_fn, img_fn))
        elif os.path.exists(mask_path_png_ext):
            paired_files.append((img_fn, mask_fn_potential_png))
        else:
            print(
                f"⚠️ Upozorenje: Maska nije pronađena za '{img_fn}' ni kao originalna ekstenzija ni kao .png. Preskačem.")

    random.shuffle(paired_files)  # Promiješaj parove slika i maski

    total_pairs = len(paired_files)
    train_count = int(total_pairs * TRAIN_RATIO)
    val_count = int(total_pairs * VAL_RATIO)
    test_count = total_pairs - train_count - val_count

    train_pairs = paired_files[:train_count]
    val_pairs = paired_files[train_count:train_count + val_count]
    test_pairs = paired_files[train_count + val_count:]

    def copy_pairs(pairs, img_dest_dir, mask_dest_dir):
        for img_fn, mask_fn in pairs:
            shutil.copy(os.path.join(ORIGINAL_IMAGES_FOLDER, img_fn), os.path.join(img_dest_dir, img_fn))
            shutil.copy(os.path.join(BINARY_MASKS_FOLDER, mask_fn), os.path.join(mask_dest_dir, mask_fn))

    print("\nKopiranje datoteka u train/val/test foldere...")
    copy_pairs(train_pairs, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    copy_pairs(val_pairs, VAL_IMAGES_DIR, VAL_MASKS_DIR)
    copy_pairs(test_pairs, TEST_IMAGES_DIR, TEST_MASKS_DIR)

    print(f"✅ Podaci podijeljeni! Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}.")
    return grain_counts_dict  # Vrati dictionary za ground truth brojanje zrna

