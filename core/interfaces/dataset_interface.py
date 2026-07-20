"""
DeepGuard — core/interfaces/dataset_interface.py

Abstract interfaces (ports) for the dataset layer following the
Dependency Inversion Principle. Infrastructure implementations
(PyTorch Datasets, file loaders) depend on these abstractions,
not the other way around.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from core.domain.entities.dataset_entity import (
    DatasetMetadataEntity,
    DatasetName,
    FaceRegionEntity,
    Label,
    SampleEntity,
    SplitName,
)
from core.domain.value_objects.dataset_split import SplitRatios


class IDatasetLoader(ABC):
    """Abstract interface for all dataset loaders.

    A dataset loader is responsible for discovering samples on disk
    and yielding SampleEntity objects. It does NOT perform any
    preprocessing or augmentation.

    Implementors: CelebDFLoader, FFPlusPlusLoader, DFDCLoader, CustomLoader.
    """

    @abstractmethod
    def load(self) -> list[SampleEntity]:
        """Discover and return all samples from the dataset root.

        Returns:
            List of SampleEntity objects representing all available samples.

        Raises:
            DatasetNotFoundError: If the root directory does not exist.
            DatasetStructureError: If the directory structure is invalid.
        """

    @abstractmethod
    def get_sample_count(self) -> dict[str, int]:
        """Return sample counts per class and total without loading all samples.

        Returns:
            Dictionary with keys 'real', 'fake', 'total'.
        """

    @abstractmethod
    def get_metadata(self) -> DatasetMetadataEntity:
        """Return dataset-level metadata.

        Returns:
            DatasetMetadataEntity with provenance and statistics.
        """

    @property
    @abstractmethod
    def dataset_name(self) -> DatasetName:
        """Return the canonical dataset identifier."""

    @property
    @abstractmethod
    def root_path(self) -> Path:
        """Return the absolute root directory of the dataset."""


class IDatasetValidator(ABC):
    """Abstract interface for dataset structure and integrity validators."""

    @abstractmethod
    def validate_structure(self, root: Path, dataset_name: DatasetName) -> list[str]:
        """Validate the directory structure of a dataset.

        Args:
            root: Root directory to validate.
            dataset_name: Expected dataset type.

        Returns:
            List of validation error messages. Empty list = valid.
        """

    @abstractmethod
    def validate_integrity(
        self,
        samples: list[SampleEntity],
        checksum_file: Path | None = None,
    ) -> list[str]:
        """Verify file integrity for a list of samples.

        Args:
            samples: Samples whose files to verify.
            checksum_file: Optional path to a checksums manifest.

        Returns:
            List of integrity error messages. Empty list = valid.
        """

    @abstractmethod
    def generate_report(
        self,
        root: Path,
        dataset_name: DatasetName,
        samples: list[SampleEntity],
    ) -> dict[str, Any]:
        """Generate a comprehensive validation report.

        Args:
            root: Dataset root directory.
            dataset_name: Dataset identifier.
            samples: All loaded samples.

        Returns:
            Dictionary containing full validation results.
        """


class IFaceExtractor(ABC):
    """Abstract interface for face detection and extraction backends."""

    @abstractmethod
    def extract_faces(
        self,
        image: np.ndarray,
        margin: float = 0.3,
        min_confidence: float = 0.9,
    ) -> list[np.ndarray]:
        """Detect and crop face regions from an image array.

        Args:
            image:          BGR or RGB numpy array (H, W, C).
            margin:         Fractional margin around detected face.
            min_confidence: Minimum detection confidence threshold.

        Returns:
            List of cropped face numpy arrays. May be empty if no face found.

        Raises:
            FaceExtractionError: If extraction fails due to an internal error.
        """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        min_confidence: float = 0.9,
    ) -> list[FaceRegionEntity]:
        """Detect face bounding boxes without cropping.

        Args:
            image:          BGR or RGB numpy array.
            min_confidence: Minimum detection confidence threshold.

        Returns:
            List of FaceRegionEntity bounding boxes.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of the underlying detection backend."""


class IDatasetSplitter(ABC):
    """Abstract interface for train/val/test dataset partitioning."""

    @abstractmethod
    def split(
        self,
        samples: list[SampleEntity],
        ratios: SplitRatios,
        seed: int = 42,
    ) -> dict[SplitName, list[SampleEntity]]:
        """Partition samples into train/val/test splits.

        Args:
            samples: Complete list of dataset samples to partition.
            ratios:  Target split ratios as a SplitRatios value object.
            seed:    Random seed for reproducibility.

        Returns:
            Dictionary mapping SplitName → list of SampleEntity.

        Raises:
            DatasetSplitError: If splits cannot be created from the given samples.
        """


class IPreprocessor(ABC):
    """Abstract interface for image/video preprocessing pipelines."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to a single image array.

        Args:
            image: Input BGR or RGB numpy array.

        Returns:
            Preprocessed numpy array.
        """

    @abstractmethod
    def preprocess_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply preprocessing to a batch of images.

        Args:
            images: List of input BGR or RGB numpy arrays.

        Returns:
            List of preprocessed numpy arrays.
        """


class IDatasetStatistics(ABC):
    """Abstract interface for computing dataset statistics."""

    @abstractmethod
    def compute(self, samples: list[SampleEntity]) -> dict[str, Any]:
        """Compute comprehensive statistics for a list of samples.

        Args:
            samples: All dataset samples to analyse.

        Returns:
            Dictionary containing computed statistics.
        """


class IDataVersioner(ABC):
    """Abstract interface for dataset version tracking."""

    @abstractmethod
    def save_version(
        self,
        samples: list[SampleEntity],
        metadata: DatasetMetadataEntity,
        output_dir: Path,
    ) -> Path:
        """Persist a versioned manifest of the current dataset state.

        Args:
            samples:    All dataset samples to record.
            metadata:   Dataset-level metadata entity.
            output_dir: Directory where the manifest is saved.

        Returns:
            Path to the saved manifest file.

        Raises:
            DatasetVersionError: If the manifest cannot be written.
        """

    @abstractmethod
    def load_version(self, manifest_path: Path) -> tuple[list[SampleEntity], DatasetMetadataEntity]:
        """Load a previously versioned dataset from its manifest.

        Args:
            manifest_path: Path to the JSON manifest file.

        Returns:
            Tuple of (list of SampleEntity, DatasetMetadataEntity).

        Raises:
            DatasetVersionError: If the manifest is invalid or corrupted.
        """

    @abstractmethod
    def list_versions(self, output_dir: Path) -> list[dict[str, Any]]:
        """List all available dataset versions in a directory.

        Args:
            output_dir: Directory containing manifest files.

        Returns:
            List of version summary dictionaries, newest first.
        """
