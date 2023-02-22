from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np
from PIL.Image import Image as PILImage
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray

if TYPE_CHECKING:
    from numpy.typing import NDArray

model_runner = bentoml.pytorch.get("pytorch").to_runner()

svc = bentoml.Service(name="pytorch_demo", runners=[model_runner])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict_image(f: PILImage) -> NDArray[t.Any]:
    assert isinstance(f, PILImage)
    img = TF.to_tensor(input)
    img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.unsqueeze(0)

    output_tensor = await model_runner.async_run(img)
    return to_numpy(output_tensor)
