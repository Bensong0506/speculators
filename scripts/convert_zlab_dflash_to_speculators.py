#!/usr/bin/env python3
"""Convert a z-lab / raw DFlash checkpoint into the speculators format.

z-lab's DFlashDraftModel ships only the trained tensors (fc, hidden_norm, norm,
layers.*) with its own config.json (no `speculators_model_type`), so this repo's
`from_pretrained` refuses it. This repo's DFlashDraftModel uses the SAME
trained-tensor names but additionally pulls embed_tokens / lm_head /
verifier_norm from the verifier and writes a speculators-format config. So we:
build the speculators model from the verifier (which loads those), overlay
z-lab's trained tensors on top, and save_pretrained -> a checkpoint that
`--from-pretrained` (FINETUNE_FROM) can warm-start from.

Usage:
    python3 scripts/convert_zlab_dflash_to_speculators.py \
        --src /home/models/Qwen3.5-9B-DFlash \
        --verifier /home/models/Qwen3.5-9B \
        --out /home/models/Qwen3.5-9B-DFlash-spec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="z-lab DFlash checkpoint dir")
    ap.add_argument("--verifier", required=True, help="verifier dir (e.g. /home/models/Qwen3.5-9B)")
    ap.add_argument("--out", required=True, help="output dir (speculators format)")
    args = ap.parse_args()

    from safetensors.torch import load_file  # noqa: PLC0415
    from transformers import Qwen3Config  # noqa: PLC0415

    from speculators.models.dflash.core import DFlashDraftModel  # noqa: PLC0415

    src = Path(args.src)
    zcfg = json.load(open(src / "config.json"))
    df = zcfg.get("dflash_config", {})
    tli = df.get("target_layer_ids") or zcfg.get("aux_hidden_state_layer_ids")
    mask = df.get("mask_token_id", zcfg.get("mask_token_id"))
    if not tli:
        raise SystemExit("could not find target_layer_ids in the source config.json")

    # The draft transformer config: num_hidden_layers = number of DRAFT layers.
    tl = Qwen3Config(
        vocab_size=zcfg["vocab_size"],
        hidden_size=zcfg["hidden_size"],
        intermediate_size=zcfg["intermediate_size"],
        num_hidden_layers=zcfg["num_hidden_layers"],
        num_attention_heads=zcfg["num_attention_heads"],
        num_key_value_heads=zcfg["num_key_value_heads"],
        head_dim=zcfg.get("head_dim"),
        hidden_act=zcfg.get("hidden_act", "silu"),
        max_position_embeddings=zcfg.get("max_position_embeddings", 262144),
        rms_norm_eps=zcfg.get("rms_norm_eps", 1e-6),
        rope_theta=zcfg.get("rope_theta", 1e7),
        attention_bias=zcfg.get("attention_bias", False),
        tie_word_embeddings=False,
    )

    print(
        f"Building speculators DFlash (block_size={zcfg['block_size']}, aux={tli}, "
        f"mask={mask}, full vocab={zcfg['vocab_size']}) + loading verifier weights from {args.verifier} ..."
    )
    model = DFlashDraftModel.from_training_args(
        verifier_config=tl,
        draft_vocab_size=zcfg["vocab_size"],   # full vocab (no draft mapping)
        block_size=zcfg["block_size"],
        max_anchors=3072,
        target_layer_ids=list(tli),
        mask_token_id=mask,
        verifier_name_or_path=args.verifier,
        sliding_window_non_causal=False,
    )

    zsd = load_file(str(src / "model.safetensors"))
    result = model.load_state_dict(zsd, strict=False)
    unexpected = list(result.unexpected_keys)
    print(f"Overlaid {len(zsd)} z-lab tensors. unexpected_keys={unexpected}")
    if unexpected:
        print("WARNING: some z-lab keys did not map onto the model — review before training:")
        for k in unexpected:
            print("   ", k)
    else:
        print("OK: every z-lab tensor mapped onto a model parameter.")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"\nSaved speculators-format DFlash -> {args.out}")
    print(f"Warm-start now loads weights:  FINETUNE_FROM={args.out}")


if __name__ == "__main__":
    main()
