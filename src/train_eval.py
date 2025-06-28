# src/train_eval.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import cv2  # Za brojanje povezanih komponenti (zrna)
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate=0.001):
    """
    Trenira segmentacijski model.

    Args:
        model (nn.Module): Model za treniranje.
        train_loader (DataLoader): DataLoader za trening podatke.
        val_loader (DataLoader): DataLoader za validacijske podatke.
        device (torch.device): Uređaj (CPU/GPU) na kojem se trenira.
        num_epochs (int): Broj epoha za treniranje.
        learning_rate (float): Stopa učenja za optimizator.

    Returns:
        tuple: Liste train_losses, val_losses, val_ious.
    """
    criterion = nn.BCEWithLogitsLoss()  # Idealno za binarnu segmentaciju

    # Treniramo samo dekoder jer je DINOv2 backbone zamrznut.
    # Ako želiš fine-tuning cijelog modela, promijeni ovo u model.parameters()
    optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_ious = []

    print("\nPočinje trening segmentacijskog modela...")
    for epoch in range(num_epochs):
        # --- Trening faza ---
        model.train()  # Postavi model u trening mod
        running_loss = 0.0
        # U DataLoaderu sad dobijamo i ime_datoteke, ali ga ne koristimo za trening
        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # --- Validaciona faza ---
        model.eval()  # Postavi model u evaluacioni mod
        val_running_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            # I ovdje dobijamo ime_datoteke, ali ga ne koristimo
            for images, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item()

                predicted_masks = torch.sigmoid(outputs)
                predicted_masks = (predicted_masks > 0.5).float()

                # IoU (Intersection over Union)
                intersection = (predicted_masks * masks).sum(dim=[1, 2, 3])
                union = (predicted_masks + masks - predicted_masks * masks).sum(dim=[1, 2, 3])
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_iou += iou.mean().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_iou = total_iou / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_ious.append(epoch_val_iou)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val IoU: {epoch_val_iou:.4f}")

    print("\nTrening završen!")
    return train_losses, val_losses, val_ious


def evaluate_model(model, test_loader, device, grain_counts_ground_truth):
    """
    Evaluira model na test setu i računa metriku brojanja zrna.

    Args:
        model (nn.Module): Obučeni model.
        test_loader (DataLoader): DataLoader za test podatke.
        device (torch.device): Uređaj (CPU/GPU) na kojem se evaluira.
        grain_counts_ground_truth (dict): Dictionary sa stvarnim brojem zrna po imenu slike.

    Returns:
        tuple: Prosječni IoU, MAE za brojanje zrna, R2 za brojanje zrna.
    """
    model.eval()  # Postavi model u evaluacioni mod
    test_iou_scores = []
    all_true_counts = []
    all_predicted_counts = []

    print("\nEvaluacija na test setu...")
    with torch.no_grad():
        for images, masks, image_names in tqdm(test_loader, desc="Evaluating on Test Set"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted_masks_probs = torch.sigmoid(outputs)
            predicted_masks_binary = (predicted_masks_probs > 0.5).float()

            # Izračunaj IoU za batch
            intersection = (predicted_masks_binary * masks).sum(dim=[1, 2, 3])
            union = (predicted_masks_binary + masks - predicted_masks_binary * masks).sum(dim=[1, 2, 3])
            batch_iou = (intersection + 1e-6) / (union + 1e-6)
            test_iou_scores.extend(batch_iou.cpu().numpy())

            # Brojanje zrna za svaku sliku u batchu
            for i in range(images.shape[0]):
                img_name_with_ext = image_names[i]
                # Neka imena datoteka u CSV-u nemaju ekstenziju
                img_name_without_ext = os.path.splitext(img_name_with_ext)[0]

                # Konvertuj maske u NumPy za OpenCV
                # Masks su već 0.0 ili 1.0 float, pretvori u uint8 za OpenCV (0 ili 255)
                true_mask_np = (masks[i, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
                predicted_mask_np = (predicted_masks_binary[i, 0, :, :].cpu().numpy() * 255).astype(np.uint8)

                # Broj povezanih komponenti (zrno) na STVARNOJ masci
                # cv2.connectedComponents vraća broj labela, uključujući pozadinu (labela 0)
                num_labels_true, _ = cv2.connectedComponents(true_mask_np)
                true_grain_count = num_labels_true - 1  # Oduzmi pozadinu (labela 0)

                # Dodaj stvarni broj zrna iz ground_truth CSV-a
                # Koristimo dictionary lookup jer je to "source of truth" za brojanje
                if img_name_without_ext in grain_counts_ground_truth:
                    all_true_counts.append(grain_counts_ground_truth[img_name_without_ext])
                else:
                    # Ovo se ne bi trebalo desiti ako su svi podaci sinkronizovani
                    print(
                        f"Upozorenje: Broj zrna za '{img_name_without_ext}' nije pronađen u CSV-u. Koristim broj iz maske.")
                    all_true_counts.append(true_grain_count)

                # Broj povezanih komponenti (zrno) na PREDVIĐENOJ masci
                num_labels_pred, _ = cv2.connectedComponents(predicted_mask_np)
                predicted_grain_count = num_labels_pred - 1
                all_predicted_counts.append(predicted_grain_count)

    # Izračunavanje ukupnih metrika
    avg_test_iou = np.mean(test_iou_scores)

    # Osiguraj da su liste iste dužine prije izračunavanja MAE/R2
    if len(all_true_counts) != len(all_predicted_counts):
        print("Upozorenje: Liste stvarnih i predviđenih brojeva nisu iste dužine. Može doći do greške.")
        min_len = min(len(all_true_counts), len(all_predicted_counts))
        all_true_counts = all_true_counts[:min_len]
        all_predicted_counts = all_predicted_counts[:min_len]

    mae_counts = mean_absolute_error(all_true_counts, all_predicted_counts)
    r2_counts = r2_score(all_true_counts, all_predicted_counts)

    return avg_test_iou, mae_counts, r2_counts


def plot_metrics(train_losses, val_losses, val_ious, num_epochs):
    """
    Vizualizira gubitak i IoU tokom treninga.
    """
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_ious, label='Validation IoU', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, data_loader, device, num_examples=3):
    """
    Prikazuje nekoliko primjera originalne slike, stvarne maske i predviđene maske.
    """
    model.eval()
    examples_shown = 0
    with torch.no_grad():
        for images, masks, image_names in data_loader:
            if examples_shown >= num_examples:
                break

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted_masks_probs = torch.sigmoid(outputs)
            predicted_masks_binary = (predicted_masks_probs > 0.5).float()

            for j in range(images.shape[0]):
                if examples_shown >= num_examples:
                    break

                img_name_with_ext = image_names[j]
                img_name_without_ext = os.path.splitext(img_name_with_ext)[0]

                # Denormalizacija slike za prikaz
                original_img_np = images[j].permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                original_img_np = std * original_img_np + mean
                original_img_np = np.clip(original_img_np, 0, 1)

                # Stvarna maska
                true_mask_np = (masks[j, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
                num_labels_true, _ = cv2.connectedComponents(true_mask_np)
                true_count = num_labels_true - 1

                # Predviđena maska
                predicted_mask_np = (predicted_masks_binary[j, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
                num_labels_pred, _ = cv2.connectedComponents(predicted_mask_np)
                predicted_count = num_labels_pred - 1

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(original_img_np)
                plt.title("Originalna Slika")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(true_mask_np, cmap='gray')
                plt.title(f"Stvarna Maska (Zrna: {true_count})")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(predicted_mask_np, cmap='gray')
                plt.title(f"Predviđena Maska (Zrna: {predicted_count})")
                plt.axis('off')
                plt.show()
                examples_shown += 1
            if examples_shown >= num_examples:
                break