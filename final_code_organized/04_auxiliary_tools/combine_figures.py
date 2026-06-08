from PIL import Image
import os

src1 = r"D:\field_batch_output_compressed_air\outputs\inference\xy_model_reasoning\pattern_and_absorption.png"
src2 = r"D:\field_batch_output_compressed_air\outputs\inference\xy_model_reasoning\xy_field_components_peak.png"
dst  = r"D:\field_batch_output_compressed_air\paper\figures\generated\ch5_field_maps.png"

img1 = Image.open(src1).convert("RGB")
img2 = Image.open(src2).convert("RGB")

# Resize both to the same width (use img2's width as reference since it's wider)
target_w = max(img1.width, img2.width)

def resize_to_width(img, w):
    ratio = w / img.width
    return img.resize((w, int(img.height * ratio)), Image.LANCZOS)

img1r = resize_to_width(img1, target_w)
img2r = resize_to_width(img2, target_w)

gap = 20
combined_h = img1r.height + gap + img2r.height
combined = Image.new("RGB", (target_w, combined_h), (255, 255, 255))
combined.paste(img1r, (0, 0))
combined.paste(img2r, (0, img1r.height + gap))

combined.save(dst, dpi=(300, 300))
print(f"Saved: {dst}  ({combined.width}x{combined.height})")
