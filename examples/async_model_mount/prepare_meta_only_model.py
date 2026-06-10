#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Prepare a metadata-only model directory for async model mount testing."""

import argparse
import shutil
from pathlib import Path

WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".msgpack",
    ".pt",
    ".pth",
    ".safetensors",
}
WEIGHT_NAME_MARKERS = {
    "model-",
    "pytorch_model",
    "quant_model_weights-",
}


def _is_weight_file(path: Path) -> bool:
    if path.suffix in WEIGHT_SUFFIXES:
        return True
    return any(path.name.startswith(marker) for marker in WEIGHT_NAME_MARKERS)


def prepare_meta_only_model(source: Path, target: Path, overwrite: bool) -> None:
    if not source.is_dir():
        raise NotADirectoryError(f"Source model directory does not exist: {source}")
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists, pass --overwrite to replace it: {target}")
        shutil.rmtree(target)

    target.mkdir(parents=True)

    for source_path in source.rglob("*"):
        relative_path = source_path.relative_to(source)
        target_path = target / relative_path

        if source_path.is_dir():
            target_path.mkdir(exist_ok=True)
            continue

        if _is_weight_file(source_path):
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Full model directory.")
    parser.add_argument("--target", type=Path, required=True, help="Metadata-only output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the target directory if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_meta_only_model(args.source, args.target, args.overwrite)
    print(f"Metadata-only model directory created at: {args.target}")


if __name__ == "__main__":
    main()
