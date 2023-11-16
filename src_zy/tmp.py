"""
python -m pip install -e .

https://github.com/yaokeepmoving/mmpretrain/tree/dev_zy/configs/otter
"""

import torch
from mmpretrain import get_model, inference_model

model = get_model('otter-9b_3rdparty_caption', pretrained=True,
                  device='cuda', generation_cfg=dict(max_new_tokens=50))
out = inference_model(model, 'demo/cat-dog.png')
print(out)
# {'pred_caption': 'The image features two adorable small puppies sitting next to each other on the grass. One puppy is brown and white, while the other is tan and white. They appear to be relaxing outdoors, enjoying each other'}
