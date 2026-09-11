"""usfeat - ultrasound feature extraction pipeline.

Extracts multiple independent feature families (PyRadiomics, DINOv2, DINOv3,
BiomedCLIP, SigLIP, ImageNet baselines) from 2D ultrasound images,
both whole-image and per-ROI, and writes one Parquet file per
(feature family x ROI) with study metadata embedded.
"""

__version__ = "0.1.0"
