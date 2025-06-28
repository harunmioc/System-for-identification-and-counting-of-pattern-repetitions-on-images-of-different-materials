# main.py

import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image

# Uvezi funkcije i klase iz tvojih modula
from src.setup_data_folders import setup_data_folders
from src.dataset import GrainSegmentationDataset
from src.model import DinoV2SegmentationModel
from src.train_eval import train_model, evaluate_model, plot_metrics, visualize_predictions # <-- DODANO OVDJE!

# --- Konfiguracija i putanje ---
# Putanje do tvojih osnovnih foldera (relativno na main.py)
ORIGINAL_IMAGES_FOLDER = "Images"
BINARY_MASKS_FOLDER = "Binary Masks"
GRAIN_COUNT_CSV = "grain_counts.csv"
# Putanje za generisani dataset split
DATASET_SPLIT_BASE = "dataset_split_for_dinov2"
CHECKPOINTS_DIR = "checkpoints"

# Kreiraj checkpoint direktorij ako ne postoji
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Postavi uređaj (GPU/CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Koristim Apple Silicon GPU (MPS).")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Koristim NVIDIA GPU (CUDA).")
else:
    device = torch.device("cpu")
    print("Koristim CPU.")

# --- 1. Priprema podataka ---
# Provjera da li su podaci već podijeljeni
# Provjeravamo postojanje 'train' foldera unutar DATASET_SPLIT_BASE
if not os.path.exists(os.path.join(DATASET_SPLIT_BASE, "train")):
    print("Pokrećem pripremu podataka...")
    # Funkcija setup_data_folders ne prima argumente, ona koristi globalne konstante unutar svog fajla
    grain_counts_ground_truth = setup_data_folders()
else:
    print("Podaci su već podijeljeni. Preskačem pripremu.")
    # Učitaj ground truth counts ako su podaci već generisani i funkcija setup_data_folders nije pozvana
    grain_counts_df = pd.read_csv(GRAIN_COUNT_CSV)
    grain_counts_ground_truth = {row['image_name']: row['grain_count'] for index, row in grain_counts_df.iterrows()}


# --- 2. Inicijalizacija Dataseta i DataLoadera ---
print("\nInicijalizacija Dataseta i DataLoadera...")

# Definiranje transformacija
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

# Kreiranje Dataset objekata
train_dataset = GrainSegmentationDataset(os.path.join(DATASET_SPLIT_BASE, "train", "images"),
                                         os.path.join(DATASET_SPLIT_BASE, "train", "masks"),
                                         image_transform=image_transform, mask_transform=mask_transform)
val_dataset = GrainSegmentationDataset(os.path.join(DATASET_SPLIT_BASE, "val", "images"),
                                       os.path.join(DATASET_SPLIT_BASE, "val", "masks"),
                                       image_transform=image_transform, mask_transform=mask_transform)
test_dataset = GrainSegmentationDataset(os.path.join(DATASET_SPLIT_BASE, "test", "images"),
                                        os.path.join(DATASET_SPLIT_BASE, "test", "masks"),
                                        image_transform=image_transform, mask_transform=mask_transform)

batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


# --- 3. Inicijalizacija Modela ---
print("\nInicijalizacija modela...")
model = DinoV2SegmentationModel(pretrained_model_name="facebook/dinov2-small").to(device)
print(f"Model je inicijaliziran i premješten na: {device}")


# --- 4. Trening Modela ---
print("\nPokrećem trening modela...")
num_epochs = 20 # Možeš prilagoditi broj epoha
train_losses, val_losses, val_ious = train_model(model, train_loader, val_loader, device, num_epochs)

# Sačuvaj model
torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "final_segmentation_model.pth"))
print(f"Model sačuvan u: {os.path.join(CHECKPOINTS_DIR, 'final_segmentation_model.pth')}")

# Vizualizuj metrike treninga
plot_metrics(train_losses, val_losses, val_ious, num_epochs)


# --- 5. Evaluacija Modela ---
print("\nPokrećem evaluaciju modela na test setu...")
avg_test_iou, mae_counts, r2_counts = evaluate_model(model, test_loader, device, grain_counts_ground_truth)

print(f"\n--- Konačni rezultati na test setu ---")
print(f"Prosječni IoU za segmentaciju: {avg_test_iou:.4f}")
print(f"Mean Absolute Error (MAE) za brojanje zrna: {mae_counts:.2f}")
print(f"R-squared (R2) za brojanje zrna: {r2_counts:.4f}")

# Vizualizacija primjera predviđanja
print("\n--- Primjeri originalne slike, stvarne i predviđene maske ---")
visualize_predictions(model, test_loader, device, num_examples=5) # Prikazati 5 primjera

print("\nProces završen!")