# convert_masks_to_png.py
import os
from PIL import Image

BINARY_MASKS_FOLDER = "../Binary Masks"  # Provjeri da je ovo putanja ispravna

print(f"Tražim JPG maske u folderu: {BINARY_MASKS_FOLDER}")
count_converted = 0

for filename in os.listdir(BINARY_MASKS_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        full_path_jpg = os.path.join(BINARY_MASKS_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]

        # Ovdje možeš odabrati kako ćeš imenovati PNG fajl:
        # Opcija 1: Samo promijeni ekstenziju u .png
        # new_filename_png = f"{base_name}.png"

        # Opcija 2: Dodaj _seg.png sufiks ako ga već nema
        if not base_name.endswith('_seg'):
            new_filename_png = f"{base_name}_seg.png"
        else:
            new_filename_png = f"{base_name}.png"  # ako već ima _seg, samo promijeni ekstenziju

        full_path_png = os.path.join(BINARY_MASKS_FOLDER, new_filename_png)

        try:
            img = Image.open(full_path_jpg).convert("L")  # Učitaj kao grayscale
            img.save(full_path_png)
            print(f"Konvertovano: {filename} -> {new_filename_png}")
            # Opcionalno: Obriši originalni JPG fajl nakon konverzije ako si siguran
            # os.remove(full_path_jpg)
            # print(f"Obrisano: {filename}")
            count_converted += 1
        except Exception as e:
            print(f"Greška pri konverziji {filename}: {e}")

print(f"\nZavršeno. Konvertovano {count_converted} JPG maski u PNG.")