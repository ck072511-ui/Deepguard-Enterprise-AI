"""
DeepGuard — scripts/prepare_dataset.py

Production-ready dataset preprocessing orchestrator script.
Processes raw images and videos to extract, align, filter, and cache face crops.
Supports parallel processing, version control manifests, statistics calculation,
and distribution plotting.
"""

from __future__ import annotations

import argparse
import logging
import logging.config
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# Adjust Python path to import from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.domain.entities.dataset_entity import (
    DatasetMetadataEntity,
    DatasetName,
    FaceRegionEntity,
    Label,
    MediaType,
    SampleEntity,
    SplitName,
)
from core.domain.value_objects.dataset_split import SplitRatios
from datasets.loaders.celeb_df_loader import CelebDFLoader
from datasets.loaders.custom_loader import CustomLoader
from datasets.loaders.dfdc_loader import DFDCLoader
from datasets.loaders.ff_plus_plus_loader import FFPlusPlusLoader
from datasets.preprocessors.video_preprocessor import FrameSamplingStrategy, VideoPreprocessor
from datasets.splitter import StratifiedSplitter, SubjectAwareSplitter
from datasets.statistics import DatasetStatistics
from datasets.versioning import DatasetVersioner
from datasets.visualization import (
    plot_class_distribution,
    plot_manipulation_distribution,
    plot_split_distribution,
)
from vision.face_extraction.alignment import FaceAligner

logger = logging.getLogger("deepguard.prepare_dataset")

# Global lazy worker face extractor (initialized inside subprocesses)
_worker_face_extractor = None


def setup_logging(config_path: str = "configs/logging_config.yaml") -> None:
    """Initialize logging config from YAML or fallback to basic config."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        try:
            with cfg_file.open() as f:
                cfg = yaml.safe_load(f)
            logging.config.dictConfig(cfg)
        except Exception as exc:
            print(f"Failed to load logging config: {exc}", file=sys.stderr)
            logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def get_face_extractor(use_face_extraction: bool, image_size: int) -> Any | None:
    """Get or initialize the face extractor on the worker process (CPU only)."""
    global _worker_face_extractor
    if not use_face_extraction:
        return None
    if _worker_face_extractor is None:
        from datasets.preprocessors.face_extractor import FaceExtractor

        # Force CPU device to avoid PyTorch/CUDA multiprocessing context conflicts
        _worker_face_extractor = FaceExtractor(
            backend="auto",
            device="cpu",
            keep_largest_only=False,
            fallback_to_full=True,
            min_face_size=40,
            image_size=image_size,
        )
    return _worker_face_extractor


def process_single_file(
    sample: SampleEntity,
    dataset_name: str,
    output_crops_dir: Path,
    use_face_extraction: bool,
    face_margin: float,
    min_confidence: float,
    image_size: int,
    sampling_strategy: str,
    max_frames: int,
    apply_clahe: bool,
    fallback_to_full: bool = True,
) -> list[SampleEntity]:
    """Preprocess a single raw image or video sample.

    Extracts frames, detects faces, aligns them, enhances contrast, and saves
    the crops to the output directory. Executed within process pool workers.
    """
    processed_samples: list[SampleEntity] = []
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if not sample.path.exists():
        logger.warning("File not found: '%s'", sample.path)
        return []

    is_video = sample.path.suffix.lower() in video_extensions

    try:
        # Step 1: Read raw frame(s)
        frames: list[tuple[int, np.ndarray]] = []  # List of (frame_index, frame_array)

        if is_video:
            # Replicate video preprocessor to get indices and raw full-res frames
            preprocessor = VideoPreprocessor(
                face_extractor=None,
                strategy=FrameSamplingStrategy(sampling_strategy),
                max_frames=max_frames,
                image_size=image_size,
            )
            # Fetch frame indices
            cap = cv2.VideoCapture(str(sample.path))
            if not cap.isOpened():
                logger.error("Cannot open video file: '%s'", sample.path)
                return []
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = preprocessor._compute_frame_indices(total_frames)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append((idx, rgb))
            finally:
                cap.release()
        else:
            # Image sample
            bgr = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.error("Cannot read image file: '%s'", sample.path)
                return []
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append((0, rgb))

        if not frames:
            return []

        # Initialize detector and aligner
        extractor = get_face_extractor(use_face_extraction, image_size)
        aligner = FaceAligner(output_size=image_size)

        # Step 2: Process each frame
        for frame_idx, frame in frames:
            h, w = frame.shape[:2]

            # Face extraction
            if use_face_extraction and extractor is not None:
                regions = extractor.detect(frame, min_confidence=min_confidence)
                if not regions and fallback_to_full:
                    regions = [
                        FaceRegionEntity(
                            x=0, y=0, width=w, height=h, confidence=1.0
                        )
                    ]
            else:
                # Bypass detection, wrap entire frame as a region
                regions = [
                    FaceRegionEntity(
                        x=0, y=0, width=w, height=h, confidence=1.0
                    )
                ]

            if not regions:
                continue

            for face_idx, region in enumerate(regions):
                # Bounding box coordinates
                x1, y1, x2, y2 = region.to_xyxy()

                # Face alignment (if 5-point landmarks are detected)
                processed_img = None
                if region.landmarks and len(region.landmarks) >= 5:
                    try:
                        # Construct standard landmark template coordinates
                        landmarks_array = np.zeros((5, 2), dtype=np.float32)
                        landmarks_array[0] = region.landmarks["left_eye"]
                        landmarks_array[1] = region.landmarks["right_eye"]
                        landmarks_array[2] = region.landmarks["nose"]
                        landmarks_array[3] = region.landmarks.get(
                            "mouth_left", region.landmarks.get("left_mouth")
                        )
                        landmarks_array[4] = region.landmarks.get(
                            "mouth_right", region.landmarks.get("right_mouth")
                        )
                        processed_img = aligner.align(frame, landmarks_array)
                    except Exception as exc:
                        logger.debug("Alignment failed for face: %s", exc)

                # Fallback to cropping with margin
                if processed_img is None:
                    expanded = region.with_margin(face_margin, image_w=w, image_h=h)
                    ex1, ey1, ex2, ey2 = expanded.to_xyxy()
                    crop = frame[ey1:ey2, ex1:ex2]
                    if crop.size == 0:
                        continue
                    processed_img = cv2.resize(
                        crop, (image_size, image_size), interpolation=cv2.INTER_CUBIC
                    )

                # Sharpness quality filter
                gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < 5.0:  # Skip extremely blurry frames
                    logger.debug(
                        "Discarded low-quality crop (sharpness=%.1f) for %s",
                        sharpness,
                        sample.path.name,
                    )
                    continue

                # Contrast enhancement (CLAHE)
                if apply_clahe:
                    lab = cv2.cvtColor(processed_img, cv2.COLOR_RGB2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l_eq = clahe_obj.apply(l_channel)
                    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
                    processed_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

                # Save output crop
                split_dir = output_crops_dir / str(sample.split)
                split_dir.mkdir(parents=True, exist_ok=True)

                filename = (
                    f"{sample.video_id or sample.path.stem}_"
                    f"{sample.sample_id[:8]}_f{frame_idx}_face{face_idx}.jpg"
                )
                dest_path = split_dir / filename

                cv2.imwrite(
                    str(dest_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                )

                # Create preprocessed SampleEntity
                new_sample = SampleEntity(
                    sample_id=f"{sample.sample_id}_f{frame_idx}_face{face_idx}",
                    path=dest_path,
                    label=sample.label,
                    dataset_name=DatasetName(dataset_name),
                    media_type=MediaType.FRAME,
                    manipulation=sample.manipulation,
                    compression=sample.compression,
                    subject_id=sample.subject_id,
                    video_id=sample.video_id or sample.path.stem,
                    frame_index=frame_idx,
                    split=sample.split,
                    metadata={
                        "original_sample_id": sample.sample_id,
                        "face_index": face_idx,
                        "confidence": region.confidence,
                        "bbox": [region.x, region.y, region.width, region.height],
                        "sharpness": round(sharpness, 2),
                    },
                )
                processed_samples.append(new_sample)

    except Exception as exc:
        logger.error("Failed processing file '%s': %s", sample.path.name, exc, exc_info=True)

    return processed_samples


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="DeepGuard Dataset Preprocessing Orchestrator"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[d.value for d in DatasetName],
        help="Dataset to process (celeb-df, ff++, dfdc, custom)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Specific split to process or 'all'",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="c23",
        help="FF++ compression level (c0, c23, c40)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of raw samples to process (for debugging)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--no-face-extraction",
        action="store_true",
        help="Bypass face detection/alignment, resize full frames directly",
    )
    parser.add_argument(
        "--config-dataset",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to dataset YAML config file",
    )

    args = parser.parse_args()

    # Load configurations
    with open(args.config_dataset) as f:
        config = yaml.safe_load(f)

    preprocessing_cfg = config.get("preprocessing", {})
    image_cfg = preprocessing_cfg.get("image", {})
    video_cfg = preprocessing_cfg.get("video", {})
    face_cfg = preprocessing_cfg.get("face_detection", {})
    processed_cfg = config.get("processed", {})

    image_size = image_cfg.get("target_size", 224)
    face_margin = image_cfg.get("face_margin", 0.3)
    min_confidence = face_cfg.get("min_confidence", 0.9)
    use_face_extraction = not args.no_face_extraction
    fallback_to_full = face_cfg.get("fallback_to_full_frame", True)

    sampling_strategy = video_cfg.get("frame_sample_strategy", "uniform")
    max_frames = video_cfg.get("max_frames_per_video", 30)

    # Resolution of outputs
    processed_root = Path(processed_cfg.get("root", "./datasets/processed"))
    output_crops_dir = processed_root / processed_cfg.get("face_crops_dir", "face_crops") / args.dataset
    manifests_dir = processed_root / processed_cfg.get("manifests_dir", "manifests")
    stats_dir = processed_root / processed_cfg.get("statistics_dir", "statistics")
    viz_dir = processed_root / processed_cfg.get("visualizations_dir", "visualizations")

    for d in [output_crops_dir, manifests_dir, stats_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Resolve dataset raw root from config
    datasets_cfg = config.get("datasets", {})
    dataset_cfg = datasets_cfg.get(args.dataset, {})
    dataset_raw_root = Path(dataset_cfg.get("root", f"./datasets/raw/{args.dataset}")).resolve()

    logger.info(
        "Initializing raw dataset loader for '%s' from root: '%s'",
        args.dataset,
        dataset_raw_root,
    )

    # Instantiate the appropriate loader
    if args.dataset == DatasetName.CELEB_DF:
        loader = CelebDFLoader(dataset_raw_root)
    elif args.dataset == DatasetName.FF_PLUS_PLUS:
        loader = FFPlusPlusLoader(dataset_raw_root, compression=args.compression)
    elif args.dataset == DatasetName.DFDC:
        loader = DFDCLoader(dataset_raw_root)
    else:  # Custom
        loader = CustomLoader(
            dataset_raw_root,
            real_dir=dataset_cfg.get("structure", {}).get("real_dir", "real"),
            fake_dir=dataset_cfg.get("structure", {}).get("fake_dir", "fake"),
        )

    # Load raw sample references
    try:
        raw_samples = loader.load()
    except Exception as exc:
        logger.critical("Failed to load raw dataset: %s", exc)
        sys.exit(1)

    if not raw_samples:
        logger.error("No raw samples discovered at '%s'. Check directory structure.", dataset_raw_root)
        sys.exit(1)

    # Filter by split if specified
    if args.split != "all":
        target_split = SplitName(args.split)
        raw_samples = [s for s in raw_samples if s.split == target_split]
        logger.info("Filtered to split '%s' | samples=%d", args.split, len(raw_samples))

    # Apply limits for quick dry runs
    if args.limit is not None:
        random.seed(42)
        raw_samples = random.sample(raw_samples, min(args.limit, len(raw_samples)))
        logger.info("Limited processing to %d samples for dry-run.", len(raw_samples))

    logger.info(
        "Starting preprocessing pipeline in parallel (%d workers) for %d samples...",
        args.num_workers,
        len(raw_samples),
    )

    processed_samples: list[SampleEntity] = []

    # Parallel execution
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                sample=sample,
                dataset_name=args.dataset,
                output_crops_dir=output_crops_dir,
                use_face_extraction=use_face_extraction,
                face_margin=face_margin,
                min_confidence=min_confidence,
                image_size=image_size,
                sampling_strategy=sampling_strategy,
                max_frames=max_frames,
                apply_clahe=False,  # Can enable CLAHE based on config/CLI flag
                fallback_to_full=fallback_to_full,
            ): sample
            for sample in raw_samples
        }

        completed_count = 0
        total_count = len(raw_samples)

        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
                processed_samples.extend(result)
            except Exception as exc:
                logger.error("Worker failed for sample '%s': %s", sample.path.name, exc)
            finally:
                completed_count += 1
                if completed_count % max(1, total_count // 10) == 0 or completed_count == total_count:
                    logger.info("Progress: %d/%d (%.1f%%) completed.", completed_count, total_count, 100.0 * completed_count / total_count)

    logger.info(
        "Preprocessing complete! Total face crops extracted: %d",
        len(processed_samples),
    )

    if not processed_samples:
        logger.warning("No face crops were successfully extracted. Skipping manifest generation.")
        return

    # Split dataset if loaders did not assign splits properly (e.g. custom layout)
    # Check if we need to split
    unassigned_splits = [s for s in processed_samples if s.split == SplitName.TRAIN and not s.subject_id]
    if args.dataset == DatasetName.CUSTOM and len(unassigned_splits) == len(processed_samples):
        logger.info("Re-partitioning custom dataset crops to ensure clean train/val/test splits.")
        # Perform stratified split
        splitter = StratifiedSplitter()
        ratios = SplitRatios(
            train=dataset_cfg.get("train_split", 0.8),
            val=dataset_cfg.get("val_split", 0.1),
            test=dataset_cfg.get("test_split", 0.1),
        )
        split_dict = splitter.split(processed_samples, ratios)
        processed_samples = []
        for s_name, s_list in split_dict.items():
            for item in s_list:
                # Update split attribute
                updated_item = SampleEntity(
                    sample_id=item.sample_id,
                    path=item.path,
                    label=item.label,
                    dataset_name=item.dataset_name,
                    media_type=item.media_type,
                    manipulation=item.manipulation,
                    compression=item.compression,
                    subject_id=item.subject_id,
                    video_id=item.video_id,
                    frame_index=item.frame_index,
                    split=s_name,
                    metadata=item.metadata,
                )
                processed_samples.append(updated_item)

    # Save Version manifest
    logger.info("Saving dataset version manifest...")
    versioner = DatasetVersioner(compute_file_checksums=False)
    metadata = DatasetMetadataEntity(
        dataset_id=f"{args.dataset}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        name=DatasetName(args.dataset),
        version=dataset_cfg.get("version", "1.0"),
        root_path=output_crops_dir,
        total_samples=len(processed_samples),
        real_count=sum(1 for s in processed_samples if s.label == Label.REAL),
        fake_count=sum(1 for s in processed_samples if s.label == Label.FAKE),
        created_at=datetime.now(timezone.utc),
        description=f"Preprocessed crops from {args.dataset}",
        tags={"compression": args.compression, "image_size": str(image_size)},
    )
    manifest_path = versioner.save_version(processed_samples, metadata, manifests_dir)
    logger.info("Saved dataset version manifest to '%s'", manifest_path)

    # Compute Statistics
    logger.info("Computing dataset statistics report...")
    stat_calculator = DatasetStatistics()
    stats = stat_calculator.compute(processed_samples)
    stat_report_path = stats_dir / f"{args.dataset}_statistics.json"
    stat_calculator.export_report(stats, stat_report_path)
    try:
        print(stat_calculator.format_summary(stats))
    except UnicodeEncodeError:
        summary_str = stat_calculator.format_summary(stats)
        summary_str = summary_str.replace("═", "=").replace("─", "-")
        print(summary_str)


    # Generate Visualizations
    logger.info("Generating distribution plots...")
    plot_class_distribution(
        processed_samples,
        title=f"{args.dataset.upper()} Class Distribution",
        output_path=viz_dir / f"{args.dataset}_class_distribution.png",
    )
    plot_split_distribution(
        processed_samples,
        title=f"{args.dataset.upper()} Split Distribution",
        output_path=viz_dir / f"{args.dataset}_split_distribution.png",
    )
    if any(s.label == Label.FAKE for s in processed_samples):
        plot_manipulation_distribution(
            processed_samples,
            title=f"{args.dataset.upper()} Manipulation Distribution",
            output_path=viz_dir / f"{args.dataset}_manipulation_distribution.png",
        )
    logger.info("Plots saved to '%s'", viz_dir)


if __name__ == "__main__":
    main()
