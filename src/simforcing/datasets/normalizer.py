from typing import Literal, Dict, Annotated, Union, Any
import os
import tempfile
import torch
import json
from collections import defaultdict
import numpy as np
from simforcing.utils.logging_config import get_logger
from simforcing.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)
ConstConstStr = Annotated[
    str,
    "format: 'const_min/const_max', where const_min and const_max give the constant range",
]
NormMode = Union[Literal["min/max", "q01/q99", "z-score"], ConstConstStr]


class SingleFieldLinearNormalizer:
    std_reg = 1e-08
    range_tol = 0.0001
    output_max = 1.0
    output_min = -1.0

    def __init__(self, stats, mode: NormMode = "min/max"):
        self.stats = stats
        self.mode = mode
        if mode == "z-score":
            input_mean, input_std = (stats["mean"], stats["std"])
            scale = 1.0 / (input_std + self.std_reg)
            offset = -input_mean / (input_std + self.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = (stats["min"], stats["max"])
            elif mode == "q01/q99":
                input_min, input_max = (stats["q01"], stats["q99"])
            else:
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)
            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[
                ignore_dim
            ]
        self.scale = scale
        self.offset = offset

    def get_stats(self):
        return self.stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return x

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.offset) / self.scale
        return x


def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable_stats = convert_tensor(dataset_stats)
    directory = os.path.dirname(os.path.abspath(file_path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(file_path) + ".", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_dataset_stats_from_json(
    file_path: str, try_convert_tensor: bool = True
) -> Dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True
            first = obj[0]
            if isinstance(first, (int, float)):
                return all((isinstance(x, (int, float)) for x in obj))
            elif isinstance(first, list):
                return all((is_numeric_list(item) for item in obj))
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if try_convert_tensor:
        data = convert_back_to_tensor(data)
    data = dict_apply(
        data, lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x
    )
    return data
