import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize

model = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-1.4", trust_remote_code=True
)


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(
        torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
    )
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# prepare input
image_path = "https://farm5.staticflickr.com/4007/4322154488_997e69e4cf_z.jpg"
orig_im = io.imread(image_path)
orig_im_size = orig_im.shape[0:2]
image = preprocess_image(orig_im, model_input_size).to(device)

# inference
result = model(image)

# post process
result_image = postprocess_image(result[0][0], orig_im_size)

# save result
pil_im = Image.fromarray(result_image)
no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
orig_image = Image.open(image_path)
no_bg_image.paste(orig_image, mask=pil_im)
