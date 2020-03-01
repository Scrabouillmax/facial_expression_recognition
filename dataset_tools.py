import PIL as pl
import numpy as np
import torch
import torchvision.transforms as transforms

size = 48
emotions = 7

# =============================================================================
# Conversion pixel tensor
# =============================================================================
def pixelstring_to_numpy(string, flatten=False, integer_pixels=False):
    pixels = string.split()
    if flatten:
        out = np.array([int(i) for i in pixels])
        return out
    out = np.zeros((size, size))
    for i in range(size):
        out[i] = np.array([int(k) for k in pixels[size * i:size * (i + 1)]])

    if integer_pixels:
        return out

    return out / 255.0


def pixelstring_to_tensor_customvgg(pixels, device):
    return torch.tensor(pixelstring_to_numpy(pixels, flatten=False), dtype=torch.float32).unsqueeze_(0).to(device)


def pixelstring_batch_totensor(psb, pixelstring_to_tensor):
    out = torch.stack(tuple([pixelstring_to_tensor(string) for string in psb]))
    return out


def emotion_batch_totensor(emb, loss_mode="BCE"):
    if loss_mode == "BCE":
        return torch.stack(emb).T.float()
    else:
        return emb

# =============================================================================
# One-hot encoding
# =============================================================================

def label_to_vector(label, device=torch.device('cpu')):
    out = torch.zeros(emotions).to(device)
    out[label] = 1
    return out


# =============================================================================
# Visualisation
# =============================================================================

def string_to_pilimage(pixelstring):
    imarray = pixelstring_to_numpy(pixelstring, integer_pixels=True)
    out = pl.Image.fromarray(imarray).convert("L")
    return out


def tensor_to_pilimage(tensor, resolution=(256, 256)):
    im = transforms.ToPILImage()(tensor.unsqueeze_(0))
    im = transforms.Resize(resolution)(im)
    return im


# =============================================================================
# Pre-processing
# =============================================================================
def preprocess_batch_custom_vgg(pixelstring_batch, emotions_batch, DEVICE, with_data_aug=True, loss_mode="BCE"):
    transformations = [
        # pre-processing
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
    if with_data_aug:
        transformations = [
                              # data augmentation
                              transforms.RandomHorizontalFlip(p=0.5)
                          ] + transformations

    pre_process = transforms.Compose(transformations)

    batch = torch.stack(
        tuple([
            pre_process(string_to_pilimage(string)) for string in pixelstring_batch
        ])
    )

    groundtruth = emotion_batch_totensor(emotions_batch, loss_mode)

    return batch, groundtruth
