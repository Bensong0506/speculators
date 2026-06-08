#!/usr/bin/env python3
"""Prune a DFlash checkpoint from N draft layers to a smaller draft stack.

The intended use here is 5-layer DFlash -> 3-layer DFlash by keeping layers
0, 2, and 4. Non-layer tensors such as fc, hidden_norm, norm, embed_tokens, and
lm_head are copied unchanged. Kept layer tensors are renumbered densely:

    layers.0.* -> layers.0.*
    layers.2.* -> layers.1.*
    layers.4.* -> layers.2.*

The output checkpoint remains a normal single-file safetensors checkpoint with
an updated config.json. It is meant to be used as FINETUNE_FROM for continued
training, not as a mathematically exact replacement for the original model.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _nested_transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("transformer_layer_config")
    return nested if isinstance(nested, dict) else config


def _num_layers_from_config(config: dict[str, Any]) -> int:
    nested = _nested_transformer_config(config)
    for candidate in (
        nested.get("num_hidden_layers"),
        config.get("num_hidden_layers"),
    ):
        if candidate is not None:
            return int(candidate)
    layer_types = nested.get("layer_types") or config.get("layer_types")
    if isinstance(layer_types, list):
        return len(layer_types)
    raise SystemExit("Could not infer DFlash draft layer count from config.json")


def _update_layer_count(
    config: dict[str, Any], keep_layers: list[int], old_count: int
) -> None:
    new_count = len(keep_layers)
    nested = _nested_transformer_config(config)

    nested["num_hidden_layers"] = new_count
    # Keep a top-level copy too because the training launcher uses plain JSON
    # inspection before transformers/pydantic rehydrates the nested config.
    config["num_hidden_layers"] = new_count

    for owner in (nested, config):
        layer_types = owner.get("layer_types")
        if isinstance(layer_types, list):
            owner["layer_types"] = [layer_types[idx] for idx in keep_layers]

    config["pruned_from_num_hidden_layers"] = old_count
    config["pruned_keep_layers"] = keep_layers


def _copy_sidecar_files(src: Path, out: Path) -> None:
    for path in src.iterdir():
        if path.name in {"config.json", "model.safetensors"}:
            continue
        if path.name == "model.safetensors.index.json":
            continue
        target = out / path.name
        if target.exists():
            continue
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--keep-layers",
        type=int,
        nargs="+",
        default=[0, 2, 4],
        help="Original layer indices to keep and densely renumber. Default: 0 2 4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite config.json/model.safetensors if they already exist in --out.",
    )
    args = parser.parse_args()

    src_config = args.src / "config.json"
    src_weights = args.src / "model.safetensors"
    if not src_config.exists():
        raise SystemExit(f"Missing source config.json: {src_config}")
    if not src_weights.exists():
        raise SystemExit(f"Missing source model.safetensors: {src_weights}")

    config = _read_config(src_config)
    old_count = _num_layers_from_config(config)
    keep_layers = args.keep_layers
    if len(keep_layers) != len(set(keep_layers)):
        raise SystemExit(f"keep layers must be unique: {keep_layers}")
    if any(idx < 0 or idx >= old_count for idx in keep_layers):
        raise SystemExit(
            f"keep layers {keep_layers} are out of range for {old_count} layers"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    out_config = args.out / "config.json"
    out_weights = args.out / "model.safetensors"
    if not args.overwrite and (out_config.exists() or out_weights.exists()):
        raise SystemExit(
            f"{args.out} already contains config.json/model.safetensors. "
            "Pass --overwrite to replace them."
        )

    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    source_state = load_file(str(src_weights))
    remap = {old: new for new, old in enumerate(keep_layers)}
    pruned_state = {}
    copied_layer_keys = {old: 0 for old in keep_layers}
    dropped_layer_keys = 0

    for key, tensor in source_state.items():
        match = LAYER_RE.match(key)
        if not match:
            pruned_state[key] = tensor
            continue

        old_layer = int(match.group(1))
        suffix = match.group(2)
        if old_layer not in remap:
            dropped_layer_keys += 1
            continue

        new_key = f"layers.{remap[old_layer]}.{suffix}"
        pruned_state[new_key] = tensor
        copied_layer_keys[old_layer] += 1

    missing_kept = [
        layer_idx for layer_idx, count in copied_layer_keys.items() if count == 0
    ]
    if missing_kept:
        raise SystemExit(f"No tensor keys found for kept layer(s): {missing_kept}")

    _copy_sidecar_files(args.src, args.out)
    _update_layer_count(config, keep_layers, old_count)
    _write_config(out_config, config)
    save_file(pruned_state, str(out_weights))

    print("DFlash layer pruning complete")
    print(f"  source:      {args.src}")
    print(f"  output:      {args.out}")
    print(f"  old layers:  {old_count}")
    print(f"  keep layers: {keep_layers}")
    print(f"  new layers:  {len(keep_layers)}")
    print(f"  tensors out: {len(pruned_state)}")
    print(f"  layer keys kept by old layer: {copied_layer_keys}")
    print(f"  layer keys dropped: {dropped_layer_keys}")


if __name__ == "__main__":
    main()
