import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import re

# --- GLOBALNI PARAMETRI ZA ANALIZU ZRNA ---
MIN_AREA_THRESHOLD = 500  # Minimalna površina zrna za uključivanje u analizu


# ------------------------------------------

def calculate_iou(mask1, mask2):
    """
    Izračunava Intersection over Union (IoU) između dvije binarne maske.
    mask1 i mask2 moraju biti NumPy array-i istih dimenzija (0 ili 1).
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)  # Dodaj epsilon za stabilnost
    return iou


def calculate_dice(mask1, mask2):
    """
    Izračunava Dice koeficijent između dvije binarne maske.
    mask1 i mask2 moraju biti NumPy array-i istih dimenzija (0 ili 1).
    """
    intersection = np.logical_and(mask1, mask2).sum()
    dice = (2. * intersection + 1e-6) / (mask1.sum() + mask2.sum() + 1e-6)  # Dodaj epsilon za stabilnost
    return dice


def analyze_masks_for_grains(masks_folder, mask_type_prefix="gt", min_area_threshold=MIN_AREA_THRESHOLD):
    """
    Analizira morfološke karakteristike zrna (povezanih komponenti) u datom folderu maski.

    Args:
        masks_folder (str): Putanja do foldera s maskama.
        mask_type_prefix (str): Prefiks za kolone u DataFrame-u ('gt' za ground truth, 'pred' za predicted).
        min_area_threshold (int): Minimalna površina zrna za uključivanje u analizu.

    Returns:
        pd.DataFrame: DataFrame sa karakteristikama zrna za sve maske u folderu.
    """
    all_grain_data = []
    mask_filenames = sorted([f for f in os.listdir(masks_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"\nAnaliza zrna u maskama iz '{masks_folder}' (tip: {mask_type_prefix})...")

    for fname in tqdm(mask_filenames, desc=f"Analiza {mask_type_prefix} maski"):
        mask_path = os.path.join(masks_folder, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Upozorenje: Nije moguće učitati masku '{fname}'. Preskačem.")
            continue

        # Izvuci čisto ime slike (bez '_seg.png' ili '_predicted_mask.png')
        img_base_name_clean = os.path.splitext(fname)[0]
        img_base_name_clean = re.sub(r'_seg$|_predicted_mask$', '', img_base_name_clean)

        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

        for label in range(1, num_labels):  # preskoči pozadinu
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area_threshold:
                continue
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], \
            stats[label, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[label]
            aspect_ratio = w / h if h != 0 else 0

            mask_grain = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_grain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True) if contours else 0

            all_grain_data.append({
                "image_name": img_base_name_clean,
                f"{mask_type_prefix}_area": area,
                f"{mask_type_prefix}_perimeter": perimeter,
                f"{mask_type_prefix}_aspect_ratio": aspect_ratio,
                f"{mask_type_prefix}_centroid_x": cx,
                f"{mask_type_prefix}_centroid_y": cy,
                f"{mask_type_prefix}_bbox_x": x,
                f"{mask_type_prefix}_bbox_y": y,
                f"{mask_type_prefix}_bbox_w": w,
                f"{mask_type_prefix}_bbox_h": h
            })
    return pd.DataFrame(all_grain_data)


def perform_mask_comparison(ground_truth_masks_folder, predicted_masks_folder, output_csv_path,
                            visualization_output_folder=None, num_visualizations=5):
    """
    Uspoređuje ground truth i predviđene maske, računa metrike sličnosti
    i generiše vizualizacije.
    Dodatno, izvlači morfološke metrike zrna iz obje vrste maski.
    """
    results_iou_dice = []
    image_names_to_visualize = set()
    num_visualized_images = 0

    if visualization_output_folder:
        os.makedirs(visualization_output_folder, exist_ok=True)
        print(f"Vizualizacije će biti spremljene u: {visualization_output_folder}")

    # Lista ground truth maski
    gt_mask_filenames_full = sorted(
        [f for f in os.listdir(ground_truth_masks_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"\nUspoređivanje IoU/Dice za maske iz '{ground_truth_masks_folder}' i '{predicted_masks_folder}'...")

    for gt_full_fname in tqdm(gt_mask_filenames_full, desc="Uspoređivanje IoU/Dice"):
        img_base_name = os.path.splitext(gt_full_fname)[0]
        img_base_name_clean = re.sub(r'_seg$', '', img_base_name)

        predicted_fname = f"{img_base_name_clean}_predicted_mask.png"
        predicted_mask_path = os.path.join(predicted_masks_folder, predicted_fname)

        if not os.path.exists(predicted_mask_path):
            # print(f"Upozorenje: Predviđena maska '{predicted_fname}' ne postoji za '{gt_full_fname}'. Preskačem.")
            continue

        gt_mask_path = os.path.join(ground_truth_masks_folder, gt_full_fname)

        gt_mask_raw = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        predicted_mask_raw = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

        if gt_mask_raw is None or predicted_mask_raw is None:
            # print(f"Upozorenje: Nije moguće učitati maske za '{gt_full_fname}'. Preskačem.")
            continue

        # Resizovanje ground truth maske na veličinu predviđene maske (224x224)
        target_height, target_width = predicted_mask_raw.shape[0], predicted_mask_raw.shape[1]

        if gt_mask_raw.shape[0] != target_height or gt_mask_raw.shape[1] != target_width:
            gt_mask_resized = cv2.resize(gt_mask_raw, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask_resized = gt_mask_raw

        # Važno: osigurajte da su maske binarne (0 ili 1) za metrike
        _, gt_mask_binary = cv2.threshold(gt_mask_resized, 127, 1, cv2.THRESH_BINARY)
        _, predicted_mask_binary = cv2.threshold(predicted_mask_raw, 127, 1, cv2.THRESH_BINARY)

        if gt_mask_binary.shape != predicted_mask_binary.shape:
            print(
                f"KRITIČNO UPOZORENJE: Dimenzije maski se I DALJE ne podudaraju za '{img_base_name_clean}' nakon resizovanja. Preskačem.")
            continue

        iou = calculate_iou(gt_mask_binary, predicted_mask_binary)
        dice = calculate_dice(gt_mask_binary, predicted_mask_binary)

        results_iou_dice.append({
            "image_name": img_base_name_clean,
            "ground_truth_mask_file": gt_full_fname,
            "predicted_mask_file": predicted_fname,
            "iou": iou,
            "dice_coefficient": dice
        })

        if visualization_output_folder and num_visualized_images < num_visualizations:
            image_names_to_visualize.add(img_base_name_clean)
            num_visualized_images = len(image_names_to_visualize)

    df_iou_dice = pd.DataFrame(results_iou_dice)
    df_iou_dice.to_csv(output_csv_path.replace(".csv", "_iou_dice.csv"), index=False)  # Spremi IoU/Dice posebno
    print(f"✅ CSV fajl sa IoU/Dice rezultatima spremljen: {output_csv_path.replace('.csv', '_iou_dice.csv')}")

    print("\n--- Deskriptivna statistika IoU i Dice koeficijenata ---")
    print(df_iou_dice.describe().round(4))

    # --- Nova faza: Analiza morfoloških metrika zrna ---
    df_gt_grains = analyze_masks_for_grains(ground_truth_masks_folder, mask_type_prefix="gt",
                                            min_area_threshold=MIN_AREA_THRESHOLD)
    df_pred_grains = analyze_masks_for_grains(predicted_masks_folder, mask_type_prefix="pred",
                                              min_area_threshold=MIN_AREA_THRESHOLD)

    # Spajanje DataFrame-ova po imenu slike, ali sa zrnom ID-jem ako je potrebno.
    # Za sada, spajamo na osnovu image_name i sumiramo po slici,
    # ili uporedjujemo distribucije.

    # Prvi pristup: Deskriptivne statistike za svaku vrstu zrna
    print("\n--- Deskriptivna statistika morfoloških metrika - Ground Truth zrna ---")
    print(df_gt_grains.describe().round(2))

    print("\n--- Deskriptivna statistika morfoloških metrika - Predicted zrna ---")
    print(df_pred_grains.describe().round(2))

    # Možemo sačuvati i ove DataFrame-ove ako želimo detalje
    df_gt_grains.to_csv(output_csv_path.replace(".csv", "_gt_grains.csv"), index=False)
    df_pred_grains.to_csv(output_csv_path.replace(".csv", "_pred_grains.csv"), index=False)
    print(f"✅ CSV fajl sa GT zrno metrikama spremljen: {output_csv_path.replace('.csv', '_gt_grains.csv')}")
    print(f"✅ CSV fajl sa Predicted zrno metrikama spremljen: {output_csv_path.replace('.csv', '_pred_grains.csv')}")

    # === Vizualizacije ===
    if visualization_output_folder and image_names_to_visualize:
        print(f"\nGenerisanje {len(image_names_to_visualize)} primjera vizualizacija...")
        for img_base_name_viz in tqdm(list(image_names_to_visualize), desc="Generisanje vizualizacija"):
            gt_mask_path = os.path.join(ground_truth_masks_folder, f"{img_base_name_viz}_seg.png")
            predicted_mask_path = os.path.join(predicted_masks_folder, f"{img_base_name_viz}_predicted_mask.png")
            original_image_path = os.path.join(ground_truth_masks_folder.replace('masks', 'images'),
                                               f"{img_base_name_viz}.png")

            gt_mask_raw = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            predicted_mask_raw = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)
            original_image_raw = cv2.imread(original_image_path)

            if gt_mask_raw is None:
                print(
                    f"Upozorenje: GT maska '{gt_mask_path}' nije pronađena ili se ne može učitati za vizualizaciju. Preskačem ovu vizualizaciju.")
                continue
            if predicted_mask_raw is None:
                print(
                    f"Upozorenje: Predicted maska '{predicted_mask_path}' nije pronađena ili se ne može učitati za vizualizaciju. Preskačem ovu vizualizaciju.")
                continue

            target_height, target_width = predicted_mask_raw.shape[0], predicted_mask_raw.shape[1]

            if gt_mask_raw.shape[0] != target_height or gt_mask_raw.shape[1] != target_width:
                gt_mask_resized = cv2.resize(gt_mask_raw, (target_width, target_height),
                                             interpolation=cv2.INTER_NEAREST)
            else:
                gt_mask_resized = gt_mask_raw
            predicted_mask_resized = predicted_mask_raw

            if original_image_raw is not None:
                original_image_display = cv2.resize(original_image_raw, (target_width, target_height),
                                                    interpolation=cv2.INTER_AREA)
            else:
                original_image_display = np.zeros((target_height, target_width, 3), dtype=np.uint8) + 128
                print(
                    f"Upozorenje: Originalna slika '{original_image_path}' nije pronađena. Koristim sivu pozadinu za overlay.")

            _, gt_mask_display = cv2.threshold(gt_mask_resized, 127, 255, cv2.THRESH_BINARY)
            _, predicted_mask_display = cv2.threshold(predicted_mask_resized, 127, 255, cv2.THRESH_BINARY)

            _, gt_mask_binary_for_iou = cv2.threshold(gt_mask_resized, 127, 1, cv2.THRESH_BINARY)
            _, predicted_mask_binary_for_iou = cv2.threshold(predicted_mask_resized, 127, 1, cv2.THRESH_BINARY)
            current_iou_for_display = calculate_iou(gt_mask_binary_for_iou, predicted_mask_binary_for_iou)

            alpha = 0.5
            mixed_image = original_image_display.copy()

            gt_mask_bool = gt_mask_display > 0
            predicted_mask_bool = predicted_mask_display > 0

            if gt_mask_bool.shape != predicted_mask_bool.shape:
                print(
                    f"Kritično upozorenje: Booleanske maske za '{img_base_name_viz}' se ne podudaraju. Preskačem overlay.")
                mixed_image = original_image_display.copy()
            else:
                mixed_image[np.logical_and(gt_mask_bool, ~predicted_mask_bool)] = \
                    (1 - alpha) * mixed_image[np.logical_and(gt_mask_bool, ~predicted_mask_bool)] + alpha * np.array(
                        [255, 0, 255])

                mixed_image[np.logical_and(~gt_mask_bool, predicted_mask_bool)] = \
                    (1 - alpha) * mixed_image[np.logical_and(~gt_mask_bool, predicted_mask_bool)] + alpha * np.array(
                        [0, 0, 255])

                mixed_image[np.logical_and(gt_mask_bool, predicted_mask_bool)] = \
                    (1 - alpha) * mixed_image[np.logical_and(gt_mask_bool, predicted_mask_bool)] + alpha * np.array(
                        [255, 255, 0])

            mixed_image = mixed_image.astype(np.uint8)

            fig, axes = plt.subplots(1, 4, figsize=(18, 6))
            if original_image_display.ndim == 3 and original_image_display.shape[2] == 3:
                axes[0].imshow(cv2.cvtColor(original_image_display, cv2.COLOR_BGR2RGB))
            else:
                axes[0].imshow(original_image_display, cmap='gray')
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(gt_mask_display, cmap='gray')
            axes[1].set_title(f"Ground Truth Mask")
            axes[1].axis('off')

            axes[2].imshow(predicted_mask_display, cmap='gray')
            axes[2].set_title("Predicted Mask")
            axes[2].axis('off')

            if mixed_image.ndim == 3 and mixed_image.shape[2] == 3:
                axes[3].imshow(cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))
            else:
                axes[3].imshow(mixed_image, cmap='gray')
            axes[3].set_title("Overlay (FN:Magenta, FP:Red, TP:Cyan)")
            axes[3].axis('off')

            plt.suptitle(f"Mask Comparison for {img_base_name_viz} (IoU: {current_iou_for_display:.4f})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(visualization_output_folder, f"{img_base_name_viz}_comparison.png"))
            plt.close(fig)

    # === Grafičke Vizualizacije Morfoloških Metrika ===
    print("\nGenerisanje grafova za morfološke metrike...")

    # Histogram površina
    plt.figure(figsize=(12, 6))
    plt.hist(df_gt_grains["gt_area"], bins=30, alpha=0.7, label='Ground Truth', color='royalblue', density=True)
    plt.hist(df_pred_grains["pred_area"], bins=30, alpha=0.7, label='Predicted', color='orange', density=True)
    plt.title("Distribucija površina zrna (normalizovano)")
    plt.xlabel("Površina (pikseli)")
    plt.ylabel("Gustina")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_visualizations_folder, "area_distribution.png"))
    plt.close()

    # Histogram perimetra
    plt.figure(figsize=(12, 6))
    plt.hist(df_gt_grains["gt_perimeter"], bins=30, alpha=0.7, label='Ground Truth', color='royalblue', density=True)
    plt.hist(df_pred_grains["pred_perimeter"], bins=30, alpha=0.7, label='Predicted', color='orange', density=True)
    plt.title("Distribucija perimetara zrna (normalizovano)")
    plt.xlabel("Perimetar (pikseli)")
    plt.ylabel("Gustina")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_visualizations_folder, "perimeter_distribution.png"))
    plt.close()

    # Histogram aspect ratio
    plt.figure(figsize=(12, 6))
    plt.hist(df_gt_grains["gt_aspect_ratio"], bins=30, alpha=0.7, label='Ground Truth', color='royalblue', density=True)
    plt.hist(df_pred_grains["pred_aspect_ratio"], bins=30, alpha=0.7, label='Predicted', color='orange', density=True)
    plt.title("Distribucija Aspect Ratio zrna (normalizovano)")
    plt.xlabel("Aspect Ratio (Širina/Visina)")
    plt.ylabel("Gustina")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_visualizations_folder, "aspect_ratio_distribution.png"))
    plt.close()

    # Scatter plot: GT Area vs Predicted Area (sa matchingom po slici)
    # Ovo je malo tricky jer nemamo 1:1 mapiranje zrna.
    # Možemo uporediti UKUPNU površinu zrna po slici ili prosečne vrednosti.
    # Za sada ćemo plotati distribucije.

    # Scatter plot: Average Area (GT vs Predicted po slici)
    avg_gt_area_per_image = df_gt_grains.groupby('image_name')['gt_area'].mean().reset_index()
    avg_pred_area_per_image = df_pred_grains.groupby('image_name')['pred_area'].mean().reset_index()

    # Spojimo ih
    merged_avg_areas = pd.merge(avg_gt_area_per_image, avg_pred_area_per_image, on='image_name', how='inner')

    if not merged_avg_areas.empty:
        plt.figure(figsize=(8, 8))
        plt.scatter(merged_avg_areas['gt_area'], merged_avg_areas['pred_area'], alpha=0.7, color='purple')
        plt.plot([0, max(merged_avg_areas['gt_area'].max(), merged_avg_areas['pred_area'].max())],
                 [0, max(merged_avg_areas['gt_area'].max(), merged_avg_areas['pred_area'].max())],
                 'r--', label='Ideal')  # Linija idealnog podudaranja
        plt.title('Prosječna površina zrna po slici: Ground Truth vs Predicted')
        plt.xlabel('Prosječna GT površina zrna')
        plt.ylabel('Prosječna Predicted površina zrna')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(output_visualizations_folder, "avg_area_scatter.png"))
        plt.close()
    else:
        print("Nema dovoljno podataka za generisanje scatter plota prosječnih površina po slici.")

    # Boxplot usporedba
    # Kombinovani DataFrame za boxplot
    combined_areas = pd.DataFrame({
        'Area Type': ['Ground Truth'] * len(df_gt_grains) + ['Predicted'] * len(df_pred_grains),
        'Area': df_gt_grains['gt_area'].tolist() + df_pred_grains['pred_area'].tolist()
    })
    plt.figure(figsize=(8, 6))
    combined_areas.boxplot(column='Area', by='Area Type', figsize=(8, 6))
    plt.title('Boxplot usporedbe površina zrna')
    plt.suptitle('')  # Ukloni automatski naslov
    plt.ylabel('Površina (pikseli)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_visualizations_folder, "boxplot_area.png"))
    plt.close()

    combined_aspect_ratios = pd.DataFrame({
        'Aspect Ratio Type': ['Ground Truth'] * len(df_gt_grains) + ['Predicted'] * len(df_pred_grains),
        'Aspect Ratio': df_gt_grains['gt_aspect_ratio'].tolist() + df_pred_grains['pred_aspect_ratio'].tolist()
    })
    plt.figure(figsize=(8, 6))
    combined_aspect_ratios.boxplot(column='Aspect Ratio', by='Aspect Ratio Type', figsize=(8, 6))
    plt.title('Boxplot usporedbe Aspect Ratio zrna')
    plt.suptitle('')
    plt.ylabel('Aspect Ratio')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_visualizations_folder, "boxplot_aspect_ratio.png"))
    plt.close()


if __name__ == "__main__":
    # Definirajte putanje
    ground_truth_masks_folder = "dataset_split_for_dinov2/test/masks"
    predicted_masks_folder = "Predicted Masks"

    output_results_csv = "mask_comparison_metrics.csv"
    output_visualizations_folder = "mask_comparison_visualizations"

    perform_mask_comparison(
        ground_truth_masks_folder,
        predicted_masks_folder,
        output_results_csv,
        visualization_output_folder=output_visualizations_folder,
        num_visualizations=10
    )