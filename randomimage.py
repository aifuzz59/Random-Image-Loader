import os
import random
import hashlib
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import torch

class LoadRandomImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"input_dir": ("STRING", {"default": ""})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_random_image"

    def load_random_image(self, input_dir):
        if not input_dir:
            raise ValueError("Please provide the directory path in the node settings.")

        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        if not files:
            raise FileNotFoundError("No image files found in the specified folder.")
        image = random.choice(files)

        image_path = os.path.join(input_dir, image)
        img = Image.open(image_path)
        output_images = []
        output_masks = []

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(cls, input_dir, *args, **kwargs):
        # Disable caching by always returning a different value
        return random.random()

NODE_CLASS_MAPPINGS = {
    "LoadRandomImage": LoadRandomImage
}