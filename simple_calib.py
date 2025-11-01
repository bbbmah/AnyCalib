import numpy as np
import torch
import json
import rawpy
from PIL import Image  # the library of choice to load images

from anycalib import AnyCalib

# default_raw_params = {
#             'use_camera_wb': True,
#             'gamma': (1, 1),
#             'no_auto_bright': True,
#             'output_bps': 8,
#         }

dev = torch.device("cuda")
image_path = "/home/work/Hwang/datasets/SR-RAW/train/00043/00001.JPG"

# load input image and convert it to a (3, H, W) tensor with RGB values in [0, 1]
image = np.array(Image.open(image_path).convert("RGB"))
image = torch.tensor(image, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255

# instantiate AnyCalib according to the desired model_id. Options:
# "anycalib_pinhole": model trained with *only* perspective (pinhole) images,
# "anycalib_gen": trained with perspective, distorted and strongly distorted images,
# "anycalib_dist": trained with distorted and strongly distorted images,
# "anycalib_edit": Trained on edited (stretched and cropped) perspective images.
model = AnyCalib(model_id="anycalib_dist").to(dev)

# Alternatively, the weights can be loaded from the huggingface hub as follows:
# NOTE: huggingface_hub (https://pypi.org/project/huggingface-hub/) needs to be installed
# model = AnyCalib().from_pretrained(model_id=<model_id>).to(dev)

# predict according to the desired camera model. Implemented camera models are detailed further below.
output = model.predict(image, cam_id="simple_radial:4")
# output is a dictionary with the following key-value pairs:
# {
#      "intrinsics": (D,) tensor with the estimated intrinsics for the selected camera model,
#      "fov_field": (N, 2) tensor with the regressed FoV field by the network. N≈320^2 (resolution close to the one seen during training),
#      "tangent_coords": alias for "fov_field",
#      "rays": (N, 3) tensor with the corresponding (via the exponential map) ray directions in the camera frame (x right, y down, z forward),
#      "pred_size": (H, W) tuple with the image size used by the network. It can be used e.g. for resizing the FoV/ray fields to the original image size.
# }

print(output)

# with open('output/result.json', 'w', encoding='utf-8') as f:
#     json.dump(output, f, ensure_ascii=False, indent=4)