#!/usr/bin/env python3
"""Download every model's weights into a local cache, for baking into the image.

Run at image build time so the container needs no network and no Hugging Face
account at run time -- which matters when it runs inside a hospital network.

    python tools/fetch_weights.py --dest /opt/usfeat/weights

Set HF_TOKEN if any repo is licence-gated. The facebook/dinov3-* repos are;
the timm/*_dinov3.lvd1689m re-hosts of the same LVD-1689M weights are not, and
are what the default config uses.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path



def fetch(dest: Path, models: dict[str, str], strict: bool) -> int:
    os.environ["HF_HOME"] = str(dest)
    os.environ["USFEAT_WEIGHTS_DIR"] = str(dest)
    dest.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []

    for name, model_id in models.items():
        print(f"\n=== {name}: {model_id}")
        try:
            if name in ("dinov2", "siglip"):
                from transformers import AutoImageProcessor, AutoModel

                AutoImageProcessor.from_pretrained(model_id, cache_dir=str(dest))
                AutoModel.from_pretrained(model_id, cache_dir=str(dest))
            elif name == "dinov3":
                if model_id.startswith("timm/"):
                    import timm

                    timm.create_model(f"hf-hub:{model_id}", pretrained=True, num_classes=0)
                else:
                    from transformers import AutoImageProcessor, AutoModel

                    AutoImageProcessor.from_pretrained(model_id, cache_dir=str(dest))
                    AutoModel.from_pretrained(model_id, cache_dir=str(dest))
            elif name == "biomedclip":
                import open_clip

                open_clip.create_model_and_transforms(model_id, cache_dir=str(dest))
            elif name.startswith("imagenet:"):
                import timm

                timm.create_model(model_id, pretrained=True, num_classes=0)
            else:
                raise ValueError(f"do not know how to fetch {name}")
            print(f"    ok")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            failures.append((name, str(exc)[:200]))

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"\ncache: {dest}  ({size / 1e9:.2f} GB)")

    if failures:
        print(f"\n{len(failures)} model(s) could not be fetched:")
        for name, err in failures:
            print(f"  {name}: {err}")
        if strict:
            return 1
        print("\ncontinuing anyway (--strict to fail the build instead); the extractor "
              "for each missing model will report itself unavailable at run time.")
    return 0


# Defaults mirror usfeat.config.DeepConfig. They are duplicated rather than
# imported so that this script does not pull in the package: keeping src/ out of
# the builder stage means editing pipeline code does not invalidate the Docker
# layer that downloads several GB of model weights.
DEFAULTS = {
    "dinov2_model": "facebook/dinov2-base",
    "dinov3_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "biomedclip_model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "siglip_model": "google/siglip-base-patch16-224",
    "imagenet_backbones": [
        "resnet50.a1_in1k",
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
    ],
}


def load_deep_settings(config_path: Path | None) -> dict:
    settings = dict(DEFAULTS)
    if config_path and config_path.exists():
        import yaml

        raw = yaml.safe_load(config_path.read_text()) or {}
        settings.update({k: v for k, v in (raw.get("deep") or {}).items() if k in DEFAULTS})
    return settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("/opt/usfeat/weights"))
    ap.add_argument("--config", type=Path, help="read model ids from this config")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any model fails to download")
    args = ap.parse_args()

    deep = load_deep_settings(args.config)
    models = {
        "dinov2": deep["dinov2_model"],
        "dinov3": deep["dinov3_model"],
        "biomedclip": deep["biomedclip_model"],
        "siglip": deep["siglip_model"],
    }
    for backbone in deep["imagenet_backbones"]:
        models[f"imagenet:{backbone}"] = backbone

    return fetch(args.dest, models, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
