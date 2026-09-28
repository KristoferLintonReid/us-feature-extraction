#!/usr/bin/env python3
"""Assert that every configured extractor can actually start, offline.

Run as the last step of the Docker build. Downloading weights and *having a
working image* are not the same thing: a transient network failure mid-build
leaves a cache that looks populated but is missing models, and because the
pipeline is deliberately tolerant of an extractor that cannot start, the result
is an image that runs, exits zero, and silently produces a third of the
expected feature tables.

This catches that at build time instead of on the collaborator's machine.

    python tools/verify_image.py --config /opt/usfeat/config/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("USFEAT_SRC", "/opt/usfeat/src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("/opt/usfeat/config/default.yaml"))
    args = ap.parse_args()

    # Offline on purpose: this must prove the baked cache is sufficient, not
    # quietly re-download whatever is missing.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from usfeat.config import Config
    from usfeat.extractors import build_extractors

    cfg = Config.load(args.config)
    cfg.deep.device = "cpu"
    expected = list(cfg.extractors)

    print(f"verifying {len(expected)} extractors can start offline: {', '.join(expected)}")
    extractors = build_extractors(cfg)
    started = {e.name for e in extractors}
    for e in extractors:
        try:
            e.close()
        except Exception:  # noqa: BLE001
            pass

    missing = [name for name in expected if name not in started]
    if missing:
        print(f"\nFAILED: {len(missing)} extractor(s) could not start: {', '.join(missing)}",
              file=sys.stderr)
        print("The weight cache in this image is incomplete. Re-run the build; if it "
              "persists, check network access to huggingface.co and HF_TOKEN.",
              file=sys.stderr)
        return 1

    print(f"\nOK: all {len(started)} extractors start offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
