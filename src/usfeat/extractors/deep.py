"""Neural feature extractors: DINOv2, DINOv3, BiomedCLIP, SigLIP, ImageNet.

All five share one presentation path so the feature sets stay comparable: the
ROI is turned into an RGB uint8 array, resized to the backbone's native input
size, normalised with that backbone's own statistics, and pushed through the
frozen encoder.

Two ROI presentations are supported and can both be emitted:
  crop -- padded bounding-box crop, so the lesion fills the field of view
  mask -- full frame with everything outside the ROI zeroed, preserving context

For the "whole" ROI both presentations collapse to the same full frame, so only
one is computed and it is emitted under the plain (un-suffixed) column names.

Weights are resolved from a local cache first (WEIGHTS_DIR / HF_HOME), so an
image built with `tools/fetch_weights.py` runs with no network access at all.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..imaging import apply_mask, crop_to_mask, to_rgb_uint8
from ..logging_setup import get_logger
from ..roi import RoiView
from .base import Extractor

log = get_logger("deep")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _offline_kwargs() -> dict:
    """Point Hugging Face loads at the baked-in weight cache when there is one."""
    kwargs: dict = {}
    weights_dir = os.environ.get("USFEAT_WEIGHTS_DIR")
    if weights_dir and Path(weights_dir).is_dir():
        kwargs["cache_dir"] = weights_dir
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


class DeepExtractor(Extractor):
    """Shared plumbing: presentation, batching, pooling, column naming."""

    input_size: int = 224
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.dcfg = cfg.deep
        self.device = resolve_device(self.dcfg.device)
        self.model = None  # set by subclasses
        self._model_id = ""

    # ------------------------------------------------------- presentation

    def _presentations(self, view: RoiView) -> dict[str, np.ndarray]:
        """ROI -> {suffix: RGB uint8 array}. Suffix is '' for the sole variant."""
        source = view.image.rgb if view.image.rgb is not None else view.image.gray

        if view.is_whole:
            return {"": to_rgb_uint8(source)}

        mode = self.dcfg.roi_mode
        wanted = ["crop", "mask"] if mode == "both" else [mode]
        out: dict[str, np.ndarray] = {}

        for variant in wanted:
            suffix = f"_{variant}" if mode == "both" else ""
            if variant == "crop":
                cropped = crop_to_mask(source, view.mask, self.dcfg.bbox_padding)
                if cropped is None:
                    continue
                out[suffix] = to_rgb_uint8(cropped[0])
            else:
                masked = _apply_mask_any(source, view.mask)
                out[suffix] = to_rgb_uint8(masked)

        if not out:
            raise RuntimeError(f"could not present ROI {view.roi} to {self.name}")
        return out

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return (t - mean) / std

    # ------------------------------------------------------------- extract

    def extract(self, view: RoiView) -> dict[str, float]:
        features: dict[str, float] = {}
        for suffix, rgb in self._presentations(view).items():
            batch = self._to_tensor(rgb).to(self.device)
            with torch.inference_mode():
                vectors = self.forward(batch)
            for pool_name, vec in vectors.items():
                arr = vec.detach().float().cpu().numpy().ravel()
                if not np.isfinite(arr).all():
                    raise ValueError(f"{self.name} produced non-finite features")
                tag = f"{pool_name}{suffix}"
                for i, value in enumerate(arr):
                    features[f"{self.prefix}{tag}_{i:04d}"] = float(value)
        if not features:
            raise RuntimeError(f"{self.name} produced no features")
        return features

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return {pool_name: (1, D) tensor}. Subclasses implement."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "extractor": self.name,
            "model_id": self._model_id,
            "input_size": self.input_size,
            "roi_mode": self.dcfg.roi_mode,
            "bbox_padding": self.dcfg.bbox_padding,
            "include_patch_mean": self.dcfg.include_patch_mean,
            "device": str(self.device),
            "torch_version": torch.__version__,
        }

    def close(self) -> None:
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _apply_mask_any(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if source.ndim == 3:
        out = np.array(source, copy=True)
        out[~mask] = 0
        return out
    return apply_mask(source, mask)


# --------------------------------------------------------------------- DINO


class _HFVisionExtractor(DeepExtractor):
    """Transformers AutoModel backbones (DINOv2, DINOv3, SigLIP)."""

    def _load(self, model_id: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        kwargs = _offline_kwargs()
        self._model_id = model_id
        processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModel.from_pretrained(model_id, **kwargs).to(self.device).eval()

        size = getattr(processor, "size", None) or {}
        self.input_size = int(
            size.get("shortest_edge") or size.get("height") or self.input_size
        )
        mean = getattr(processor, "image_mean", None)
        std = getattr(processor, "image_std", None)
        if mean and std:
            self.mean, self.std = tuple(mean), tuple(std)
        log.info("%s: %s at %dpx on %s", self.name, model_id, self.input_size, self.device)

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(pixel_values=batch)
        hidden = out.last_hidden_state  # (1, tokens, D)
        pools: dict[str, torch.Tensor] = {}

        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            pools["pooled"] = pooled
        else:
            pools["cls"] = hidden[:, 0]

        if self.dcfg.include_patch_mean:
            # Skip the CLS (and any register tokens DINOv3 prepends) before pooling.
            n_special = getattr(getattr(self.model, "config", None), "num_register_tokens", 0) or 0
            patches = hidden[:, 1 + n_special:]
            if patches.shape[1] > 0:
                pools["patchmean"] = patches.mean(dim=1)
        return pools


class DINOv2Extractor(_HFVisionExtractor):
    name = "dinov2"
    prefix = "dinov2_"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._load(cfg.deep.dinov2_model)


class DINOv3Extractor(_HFVisionExtractor):
    """DINOv3, from either the timm re-host or the gated Meta repo.

    `facebook/dinov3-*` is licence-gated, which is awkward for a container that
    has to run on someone else's network. The `timm/*_dinov3.lvd1689m` repos
    carry the same LVD-1689M weights without the gate, so they are the default;
    set `deep.dinov3_model` to a `facebook/...` id to use the Meta repo once the
    licence has been accepted on the running account.
    """

    name = "dinov3"
    prefix = "dinov3_"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        model_id = cfg.deep.dinov3_model
        try:
            if model_id.startswith("timm/") or model_id.startswith("timm:"):
                self._load_timm(model_id.replace("timm:", "timm/", 1))
            else:
                self._load(model_id)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"DINOv3 weights unavailable for {model_id!r} ({type(exc).__name__}: {exc}). "
                "The facebook/dinov3-* repos are licence-gated -- either accept the licence "
                "on huggingface.co and set HF_TOKEN, or use the ungated timm re-host "
                "(deep.dinov3_model: timm/vit_base_patch16_dinov3.lvd1689m)."
            ) from exc

    def _load_timm(self, model_id: str) -> None:
        import timm

        cache = os.environ.get("USFEAT_WEIGHTS_DIR")
        if cache and Path(cache).is_dir():
            os.environ.setdefault("HF_HOME", cache)

        self._model_id = model_id
        self._timm = True
        self.model = timm.create_model(
            f"hf-hub:{model_id}", pretrained=True, num_classes=0
        ).to(self.device).eval()

        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.input_size = int(data_cfg["input_size"][-1])
        self.mean = tuple(data_cfg["mean"])
        self.std = tuple(data_cfg["std"])
        log.info("dinov3: %s at %dpx (%d-d) on %s",
                 model_id, self.input_size, self.model.num_features, self.device)

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if not getattr(self, "_timm", False):
            return super().forward(batch)

        tokens = self.model.forward_features(batch)  # (1, tokens, D)
        pools: dict[str, torch.Tensor] = {}
        if tokens.ndim == 3:
            pools["cls"] = tokens[:, 0]
            if self.dcfg.include_patch_mean:
                n_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
                patches = tokens[:, n_prefix:]
                if patches.shape[1] > 0:
                    pools["patchmean"] = patches.mean(dim=1)
        else:
            pools["pooled"] = self.model.forward_head(tokens, pre_logits=True)
        return pools


class SigLIPExtractor(_HFVisionExtractor):
    name = "siglip"
    prefix = "siglip_"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        from transformers import AutoImageProcessor, SiglipVisionModel

        model_id = cfg.deep.siglip_model
        kwargs = _offline_kwargs()
        self._model_id = model_id
        processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
        self.model = SiglipVisionModel.from_pretrained(model_id, **kwargs).to(self.device).eval()
        size = getattr(processor, "size", None) or {}
        self.input_size = int(size.get("height") or size.get("shortest_edge") or 224)
        self.mean = tuple(processor.image_mean)
        self.std = tuple(processor.image_std)
        log.info("siglip: %s at %dpx on %s", model_id, self.input_size, self.device)


# --------------------------------------------------------------- BiomedCLIP


class BiomedCLIPExtractor(DeepExtractor):
    """BiomedCLIP image tower, loaded through open_clip."""

    name = "biomedclip"
    prefix = "biomedclip_"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        import open_clip

        model_id = cfg.deep.biomedclip_model
        self._model_id = model_id
        cache = os.environ.get("USFEAT_WEIGHTS_DIR")
        kwargs = {"cache_dir": cache} if cache and Path(cache).is_dir() else {}
        model, _, preprocess = open_clip.create_model_and_transforms(model_id, **kwargs)
        self.model = model.to(self.device).eval()

        # Mirror open_clip's own preprocessing rather than guessing at it.
        self.input_size = int(getattr(model.visual, "image_size", (224, 224))[0]
                              if isinstance(getattr(model.visual, "image_size", 224), (tuple, list))
                              else getattr(model.visual, "image_size", 224))
        for t in getattr(preprocess, "transforms", []):
            if hasattr(t, "mean") and hasattr(t, "std"):
                self.mean, self.std = tuple(t.mean), tuple(t.std)
        log.info("biomedclip: %s at %dpx on %s", model_id, self.input_size, self.device)

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.model.encode_image(batch)
        return {"img": embedding}


# ---------------------------------------------------------------- ImageNet


class ImageNetExtractor(DeepExtractor):
    """Supervised-ImageNet baselines via timm, one column block per backbone."""

    name = "imagenet"
    prefix = "imagenet_"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        import timm

        self.backbones: dict[str, tuple[torch.nn.Module, tuple, tuple, int]] = {}
        cache = os.environ.get("USFEAT_WEIGHTS_DIR")
        if cache and Path(cache).is_dir():
            os.environ.setdefault("HF_HOME", cache)

        for backbone in cfg.deep.imagenet_backbones:
            model = timm.create_model(backbone, pretrained=True, num_classes=0).to(self.device).eval()
            data_cfg = timm.data.resolve_model_data_config(model)
            size = int(data_cfg["input_size"][-1])
            self.backbones[backbone] = (model, tuple(data_cfg["mean"]), tuple(data_cfg["std"]), size)
            log.info("imagenet: %s at %dpx (%d-d) on %s",
                     backbone, size, model.num_features, self.device)

        if not self.backbones:
            raise RuntimeError("no imagenet backbones configured")
        self._model_id = ",".join(self.backbones)

    def extract(self, view: RoiView) -> dict[str, float]:
        features: dict[str, float] = {}
        presentations = self._presentations(view)

        for backbone, (model, mean, std, size) in self.backbones.items():
            tag = backbone.split(".")[0].replace("-", "_")
            self.mean, self.std, self.input_size = mean, std, size
            for suffix, rgb in presentations.items():
                batch = self._to_tensor(rgb).to(self.device)
                with torch.inference_mode():
                    vec = model(batch)
                arr = vec.detach().float().cpu().numpy().ravel()
                if not np.isfinite(arr).all():
                    raise ValueError(f"imagenet/{backbone} produced non-finite features")
                for i, value in enumerate(arr):
                    features[f"{self.prefix}{tag}{suffix}_{i:04d}"] = float(value)

        if not features:
            raise RuntimeError("imagenet produced no features")
        return features

    def describe(self) -> dict:
        base = super().describe()
        base["backbones"] = list(self.backbones)
        return base

    def close(self) -> None:
        self.backbones.clear()
        super().close()


def build_deep_extractor(name: str, cfg: Config) -> DeepExtractor:
    table = {
        "dinov2": DINOv2Extractor,
        "dinov3": DINOv3Extractor,
        "siglip": SigLIPExtractor,
        "biomedclip": BiomedCLIPExtractor,
        "imagenet": ImageNetExtractor,
    }
    return table[name](cfg)
