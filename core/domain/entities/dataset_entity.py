"""
DeepGuard — core/domain/entities/dataset_entity.py

Immutable domain entities representing dataset samples, metadata,
and dataset-level records. These are pure Python dataclasses with
zero infrastructure dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Label(IntEnum):
    """Binary classification label for deepfake detection."""

    REAL = 0
    FAKE = 1

    @classmethod
    def from_string(cls, value: str) -> "Label":
        """Parse a label from a string representation.

        Args:
            value: One of 'real', 'fake', '0', '1' (case-insensitive).

        Returns:
            Corresponding Label enum member.

        Raises:
            ValueError: If value cannot be mapped to a known label.
        """
        normalised = value.strip().lower()
        mapping = {
            "real": cls.REAL,
            "0": cls.REAL,
            "original": cls.REAL,
            "fake": cls.FAKE,
            "1": cls.FAKE,
            "manipulated": cls.FAKE,
            "deepfake": cls.FAKE,
            "synthesized": cls.FAKE,
        }
        if normalised not in mapping:
            raise ValueError(
                f"Cannot parse label from '{value}'. "
                f"Accepted values: {list(mapping.keys())}"
            )
        return mapping[normalised]


class DatasetName(StrEnum):
    """Supported deepfake detection dataset identifiers."""

    CELEB_DF = "celeb-df"
    FF_PLUS_PLUS = "ff++"
    DFDC = "dfdc"
    CUSTOM = "custom"


class ManipulationType(StrEnum):
    """Specific manipulation method used to create a fake sample."""

    NONE = "none"               # Real (unmanipulated)
    DEEPFAKES = "Deepfakes"
    FACE2FACE = "Face2Face"
    FACESWAP = "FaceSwap"
    NEURAL_TEXTURES = "NeuralTextures"
    FACESHIFTER = "FaceShifter"
    UNKNOWN = "unknown"


class CompressionLevel(StrEnum):
    """FaceForensics++ compression level identifiers."""

    RAW = "c0"
    LIGHT = "c23"
    HEAVY = "c40"
    UNKNOWN = "unknown"


class MediaType(StrEnum):
    """Input media type for a sample."""

    IMAGE = "image"
    VIDEO = "video"
    FRAME = "frame"  # Extracted video frame


class SplitName(StrEnum):
    """Dataset partition names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ---------------------------------------------------------------------------
# Domain Entities
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampleEntity:
    """Represents a single labelled media sample in the dataset.

    This entity is immutable and carries all information needed to
    locate and classify a single data point.

    Attributes:
        sample_id:        Unique identifier for this sample.
        path:             Absolute path to the image/video file.
        label:            Ground-truth classification label.
        dataset_name:     Source dataset this sample belongs to.
        media_type:       Whether path points to an image or video.
        manipulation:     Deepfake manipulation method (NONE for real).
        compression:      Compression level (FF++ specific).
        subject_id:       Identity of the subject (for stratified splits).
        video_id:         Source video ID (for frame-level samples).
        frame_index:      Frame number within the source video (-1 = N/A).
        split:            Dataset partition this sample is assigned to.
        metadata:         Arbitrary extra key-value metadata.
    """

    sample_id: str
    path: Path
    label: Label
    dataset_name: DatasetName
    media_type: MediaType = MediaType.IMAGE
    manipulation: ManipulationType = ManipulationType.NONE
    compression: CompressionLevel = CompressionLevel.UNKNOWN
    subject_id: str = ""
    video_id: str = ""
    frame_index: int = -1
    split: SplitName = SplitName.TRAIN
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        path: Path | str,
        label: Label | int,
        dataset_name: DatasetName | str,
        **kwargs: Any,
    ) -> "SampleEntity":
        """Factory method that auto-generates a unique sample_id.

        Args:
            path: Path to the media file.
            label: Ground-truth label (Label enum or int 0/1).
            dataset_name: Source dataset name.
            **kwargs: Optional SampleEntity field overrides.

        Returns:
            Fully initialised SampleEntity with a generated UUID.
        """
        return cls(
            sample_id=str(uuid.uuid4()),
            path=Path(path),
            label=Label(label),
            dataset_name=DatasetName(dataset_name),
            **kwargs,
        )

    @property
    def is_real(self) -> bool:
        """Return True if this sample is labelled as real."""
        return self.label == Label.REAL

    @property
    def is_fake(self) -> bool:
        """Return True if this sample is labelled as fake."""
        return self.label == Label.FAKE

    @property
    def filename(self) -> str:
        """Return the bare filename without parent directories."""
        return self.path.name

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dictionary representation with string-typed enum values.
        """
        return {
            "sample_id": self.sample_id,
            "path": str(self.path),
            "label": int(self.label),
            "label_name": self.label.name,
            "dataset_name": str(self.dataset_name),
            "media_type": str(self.media_type),
            "manipulation": str(self.manipulation),
            "compression": str(self.compression),
            "subject_id": self.subject_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "split": str(self.split),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class DatasetMetadataEntity:
    """Dataset-level metadata record.

    Attributes:
        dataset_id:       Unique identifier for this dataset version.
        name:             Dataset name (e.g. 'ff++').
        version:          Semantic version string (e.g. '1.0.0').
        root_path:        Absolute root directory of the dataset.
        total_samples:    Total sample count.
        real_count:       Count of real samples.
        fake_count:       Count of fake samples.
        created_at:       ISO-8601 creation timestamp.
        checksum:         SHA-256 checksum of the manifest file.
        description:      Optional human-readable description.
        tags:             Arbitrary metadata tags.
    """

    dataset_id: str
    name: DatasetName
    version: str
    root_path: Path
    total_samples: int
    real_count: int
    fake_count: int
    created_at: datetime
    checksum: str = ""
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def class_balance(self) -> float:
        """Return real/(real+fake) ratio. 0.5 = perfectly balanced."""
        if self.total_samples == 0:
            return 0.0
        return self.real_count / self.total_samples

    @property
    def is_balanced(self) -> bool:
        """Return True if class imbalance is within 10% of 50/50."""
        return 0.4 <= self.class_balance <= 0.6

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "name": str(self.name),
            "version": self.version,
            "root_path": str(self.root_path),
            "total_samples": self.total_samples,
            "real_count": self.real_count,
            "fake_count": self.fake_count,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "description": self.description,
            "class_balance": round(self.class_balance, 4),
            "is_balanced": self.is_balanced,
            "tags": self.tags,
        }


@dataclass(frozen=True, slots=True)
class FaceRegionEntity:
    """Detected face bounding box within an image or frame.

    Attributes:
        x:          Left coordinate (pixels).
        y:          Top coordinate (pixels).
        width:      Bounding box width (pixels).
        height:     Bounding box height (pixels).
        confidence: Detection confidence score [0.0, 1.0].
        landmarks:  Optional facial landmark coordinates.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def area(self) -> int:
        """Return bounding box area in pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Return width/height ratio."""
        return self.width / max(self.height, 1)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) format used by OpenCV and torchvision."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def with_margin(self, margin: float, image_w: int, image_h: int) -> "FaceRegionEntity":
        """Return a new FaceRegionEntity expanded by a fractional margin.

        Args:
            margin: Fractional margin relative to bounding box size (e.g. 0.3).
            image_w: Image width in pixels (clamp boundary).
            image_h: Image height in pixels (clamp boundary).

        Returns:
            New FaceRegionEntity with expanded, clamped coordinates.
        """
        dw = int(self.width * margin)
        dh = int(self.height * margin)
        new_x = max(0, self.x - dw)
        new_y = max(0, self.y - dh)
        new_w = min(image_w - new_x, self.width + 2 * dw)
        new_h = min(image_h - new_y, self.height + 2 * dh)
        return FaceRegionEntity(
            x=new_x,
            y=new_y,
            width=new_w,
            height=new_h,
            confidence=self.confidence,
            landmarks=self.landmarks,
        )
