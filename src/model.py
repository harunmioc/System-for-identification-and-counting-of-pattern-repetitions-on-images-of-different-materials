import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


class DinoV2SegmentationModel(nn.Module):
    def __init__(self, pretrained_model_name="facebook/dinov2-small", num_classes=1):
        super().__init__()
        # Učitavanje predtreniranog DINOv2 modela
        # DINOv2-small je dobar balans između performansi i resursa
        self.dinov2 = Dinov2Model.from_pretrained(pretrained_model_name)

        # Zamrzavanje svih parametara DINOv2 modela za početak.
        # Ovo je preporučljivo u ranim fazama treninga kako bi se dekoder stabilizirao.
        # Kasnije ih možete odmrzuti za fine-tuning cijelog modela ako želite bolje performanse.
        for param in self.dinov2.parameters():
            param.requires_grad = False

        # DINOv2 modeli (poput dinov2-small) imaju 'hidden_size' (dimenziju značajki)
        # za dinov2-small to je 384. Patch size je 16.
        # Za ulaz 224x224, DINOv2 izlazi feature mapu veličine 14x14 (jer 224 / 16 = 14)
        hidden_size = self.dinov2.config.hidden_size

        # Dekoder: Jednostavna arhitektura s ConvTranspose2d slojevima za upscaling.
        # Cilj je postupno vratiti značajke na rezoluciju ulazne slike (224x224)
        # i generisati binarnu masku (num_classes=1).
        self.decoder = nn.Sequential(
            # Upscale 1: ulaz 14x14, izlaz 28x28
            # ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            nn.ConvTranspose2d(hidden_size, 256, kernel_size=4, stride=2, padding=1), # 14*2 = 28
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Upscale 2: ulaz 28x28, izlaz 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 28*2 = 56
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Upscale 3: ulaz 56x56, izlaz 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 56*2 = 112
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Upscale 4: ulaz 112x112, izlaz 224x224
            # Ovdje je problem. Za 112 -> 224 treba nam stride=2.
            # Da bismo bili sigurni da dobijemo točno 224, možemo koristiti output_padding.
            # Izlazna veličina L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
            # Za 112x112 -> 224x224, sa kernel_size=4, stride=2, padding=1:
            # L_out = (112 - 1) * 2 - 2 * 1 + 4 = 111 * 2 - 2 + 4 = 222 - 2 + 4 = 224
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, pixel_values):
        # DINOv2 vraća razne izlaze. Nama trebaju 'last_hidden_state' koji sadrži tokene.
        # `output_hidden_states=True` osigurava da ih dobijemo.
        outputs = self.dinov2(pixel_values=pixel_values, output_hidden_states=True)

        # 'last_hidden_state' je tensor oblika (batch_size, num_patches + 1, hidden_size).
        # Prvi token (indeks 0) je [CLS] token (globalna reprezentacija slike),
        # a ostali su 'patch tokens' koji predstavljaju značajke za svaki patch slike.
        # Za segmentaciju, nama trebaju patch tokens.
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # Uklanjamo [CLS] token

        # Sada moramo preoblikovati te 'patch tokens' u 2D feature mapu
        # koja se može obraditi pomoću konvolucijskih slojeva (ConvTranspose2d).
        batch_size, num_patches, hidden_size = patch_tokens.shape

        # Izračunavamo visinu i širinu feature mape. DINOv2 patchevi su kvadratni.
        h_feat = w_feat = int(num_patches ** 0.5)

        # Permutujemo dimenzije i preoblikujemo u (Batch, Channels, Height, Width) format
        # koji očekuje PyTorch konvolucijski sloj.
        features = patch_tokens.permute(0, 2, 1).reshape(batch_size, hidden_size, h_feat, w_feat)

        # Zatim proslijeđujemo značajke kroz dekoder za generisanje segmentacijske maske.
        mask_logits = self.decoder(features)

        if mask_logits.shape[2] != 224 or mask_logits.shape[3] != 224:
            start_h = (mask_logits.shape[2] - 224) // 2
            start_w = (mask_logits.shape[3] - 224) // 2
            mask_logits = mask_logits[:, :, start_h:start_h + 224, start_w:start_w + 224]


        # `mask_logits` su neobrađene izlazne vrijednosti (logiti) modela.
        # One će biti proslijeđene funkciji gubitka (`BCEWithLogitsLoss`) koja će ih
        # automatski transformirati u vjerovatnoće i izračunati gubitak.
        return mask_logits