import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from skimage import restoration, img_as_float, img_as_ubyte
from scipy import ndimage

class ImageProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process_image(self, image_path, output_path=None):
        try:
            # Cargar imagen
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Convertir a float
            img_float = img_as_float(img_array)
            
            # Aplicar Non-local Means Denoising
            denoise_fast = restoration.denoise_nl_means(
                img_float,
                patch_size=5,
                patch_distance=6,
                h=0.1,
                channel_axis=-1,  # Reemplaza multichannel=True
                fast_mode=True
            )
            
            # Aplicar TV denoising para suavizar más
            denoise_tv = restoration.denoise_tv_chambolle(
                denoise_fast,
                weight=0.1,
                channel_axis=-1  # Reemplaza multichannel=True
            )
            
            # Mejorar nitidez
            sharpened = ndimage.gaussian_gradient_magnitude(denoise_tv, sigma=1)
            denoise_sharp = denoise_tv + 0.5 * sharpened
            
            # Convertir de vuelta a uint8
            output = img_as_ubyte(denoise_sharp.clip(0, 1))
            
            # Convertir a PIL Image
            output_image = Image.fromarray(output)
            
            if output_path:
                output_image.save(output_path)
            return output_image
                
        except Exception as e:
            print(f"Error procesando imagen: {str(e)}")
            raise