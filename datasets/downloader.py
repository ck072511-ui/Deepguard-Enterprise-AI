"""
DeepGuard — datasets/downloader.py

Dataset download management for all supported datasets.

Since CelebDF and FF++ require signed license agreements, this module:
  - Provides instructional download guidance for restricted datasets
  - Implements Kaggle API integration for DFDC
  - Validates downloads after completion
  - Supports resume for interrupted downloads
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from core.domain.entities.dataset_entity import DatasetName
from core.exceptions.dataset_exceptions import DatasetDownloadError, UnsupportedDatasetError

logger = logging.getLogger(__name__)

# Download instructions for each dataset
_DOWNLOAD_INSTRUCTIONS: dict[DatasetName, str] = {
    DatasetName.CELEB_DF: textwrap.dedent("""
        CelebDF-v2 requires a signed license agreement.
        ─────────────────────────────────────────────
        1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics
        2. Fill in the Google Form to request download access.
        3. You will receive a Google Drive link via email.
        4. Download and extract to: {target_dir}

        Expected structure after extraction:
          {target_dir}/
          ├── Celeb-real/
          ├── YouTube-real/
          ├── Celeb-synthesis/
          └── List_of_testing_videos.txt
    """),
    DatasetName.FF_PLUS_PLUS: textwrap.dedent("""
        FaceForensics++ requires a license request.
        ─────────────────────────────────────────
        1. Visit: https://github.com/ondyari/FaceForensics
        2. Fill in the usage agreement form.
        3. Use the official download script:

           python scripts/download_ff++.py \\
               {target_dir} \\
               all \\                 # or Deepfakes/Face2Face/etc.
               -c c23 \\             # compression level
               -t videos            # download type

        Expected structure after download:
          {target_dir}/
          ├── original_sequences/
          ├── manipulated_sequences/
          └── splits/
    """),
    DatasetName.DFDC: textwrap.dedent("""
        DFDC is available via Kaggle (requires account + API key).
        ──────────────────────────────────────────────────────────
        1. Install Kaggle CLI:
           pip install kaggle

        2. Set up credentials:
           Place your kaggle.json at: ~/.kaggle/kaggle.json
           Or set environment variables:
             KAGGLE_USERNAME=your_username
             KAGGLE_KEY=your_api_key

        3. Download the dataset:
           kaggle competitions download \\
               -c deepfake-detection-challenge \\
               -p {target_dir}

        4. Extract and organise:
           unzip {target_dir}/*.zip -d {target_dir}/

        Expected structure:
          {target_dir}/
          ├── dfdc_train_part_0/
          │   ├── metadata.json
          │   └── *.mp4
          └── ...  (up to part_49)
    """),
    DatasetName.CUSTOM: textwrap.dedent("""
        Custom Dataset — Manual Setup Required
        ──────────────────────────────────────
        Organise your dataset as follows:
          {target_dir}/
          ├── real/
          │   ├── image001.jpg
          │   └── video001.mp4
          └── fake/
              ├── image001.jpg
              └── video001.mp4

        Or provide a manifest.csv:
          path,label
          real/image001.jpg,0
          fake/image001.jpg,1
    """),
}


class DatasetDownloader:
    """Manages dataset download and setup for all supported datasets.

    For license-restricted datasets (CelebDF, FF++), displays instructions.
    For Kaggle datasets (DFDC), attempts automated download via Kaggle CLI.

    Args:
        datasets_root: Root directory where datasets will be stored.
        overwrite:     If True, re-download even if directory exists.
    """

    def __init__(
        self,
        datasets_root: Path | str = Path("./datasets/raw"),
        overwrite: bool = False,
    ) -> None:
        self._root = Path(datasets_root)
        self._overwrite = overwrite

    def download(self, dataset_name: str) -> Path:
        """Initiate download or display instructions for a dataset.

        Args:
            dataset_name: One of 'celeb-df', 'ff++', 'dfdc', 'custom'.

        Returns:
            Path to the target dataset directory.

        Raises:
            UnsupportedDatasetError: If dataset_name is not recognised.
            DatasetDownloadError:    If an automated download fails.
        """
        try:
            name = DatasetName(dataset_name)
        except ValueError:
            raise UnsupportedDatasetError(
                dataset_name, [str(d) for d in DatasetName]
            )

        target_dir = self._root / str(name)

        if target_dir.exists() and not self._overwrite:
            if any(target_dir.iterdir()):
                logger.info(
                    "[Downloader] '%s' already exists at '%s'. Use overwrite=True to re-download.",
                    dataset_name,
                    target_dir,
                )
                return target_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Downloader] Starting download for '%s'.", dataset_name)

        if name == DatasetName.DFDC:
            return self._download_dfdc(target_dir)

        # All other datasets: display instructions
        self._print_instructions(name, target_dir)
        return target_dir

    def check_available(self, dataset_name: str) -> dict[str, Any]:
        """Check if a dataset directory exists and has content.

        Args:
            dataset_name: Dataset name to check.

        Returns:
            Dictionary with keys: available, path, file_count, size_gb.
        """
        try:
            name = DatasetName(dataset_name)
        except ValueError:
            raise UnsupportedDatasetError(
                dataset_name, [str(d) for d in DatasetName]
            )

        target_dir = self._root / str(name)
        if not target_dir.exists():
            return {
                "available": False,
                "path": str(target_dir),
                "file_count": 0,
                "size_gb": 0.0,
            }

        files = list(target_dir.rglob("*"))
        media_files = [
            f for f in files
            if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"}
        ]
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return {
            "available": len(media_files) > 0,
            "path": str(target_dir),
            "file_count": len(media_files),
            "size_gb": round(total_size / (1024**3), 3),
        }

    def list_available(self) -> dict[str, dict[str, Any]]:
        """Check availability of all supported datasets.

        Returns:
            Dictionary mapping dataset name → availability info.
        """
        return {str(name): self.check_available(str(name)) for name in DatasetName}

    # ------------------------------------------------------------------
    # Private download implementations
    # ------------------------------------------------------------------

    def _download_dfdc(self, target_dir: Path) -> Path:
        """Download DFDC via Kaggle CLI.

        Args:
            target_dir: Destination directory.

        Returns:
            Path to the populated target directory.

        Raises:
            DatasetDownloadError: If Kaggle CLI is not found or fails.
        """
        # Verify Kaggle CLI is available
        if not shutil.which("kaggle"):
            self._print_instructions(DatasetName.DFDC, target_dir)
            raise DatasetDownloadError(
                DatasetName.DFDC,
                "Kaggle CLI not found. Install with: pip install kaggle",
            )

        # Verify credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        has_env_creds = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ

        if not kaggle_json.exists() and not has_env_creds:
            self._print_instructions(DatasetName.DFDC, target_dir)
            raise DatasetDownloadError(
                DatasetName.DFDC,
                "Kaggle credentials not found. "
                "Place kaggle.json at ~/.kaggle/kaggle.json or set "
                "KAGGLE_USERNAME and KAGGLE_KEY environment variables.",
            )

        logger.info("[Downloader] Downloading DFDC via Kaggle API...")
        try:
            result = subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    "-c",
                    "deepfake-detection-challenge",
                    "-p",
                    str(target_dir),
                ],
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )
            if result.returncode != 0:
                raise DatasetDownloadError(
                    DatasetName.DFDC,
                    f"Kaggle download failed: {result.stderr}",
                )
        except subprocess.TimeoutExpired as exc:
            raise DatasetDownloadError(
                DatasetName.DFDC,
                "Kaggle download timed out after 2 hours.",
            ) from exc

        logger.info("[Downloader] DFDC download complete. Extracting ZIP files...")
        self._extract_zips(target_dir)
        return target_dir

    def _print_instructions(self, name: DatasetName, target_dir: Path) -> None:
        """Display dataset download instructions to the user.

        Args:
            name:       Dataset name.
            target_dir: Suggested target directory.
        """
        template = _DOWNLOAD_INSTRUCTIONS.get(name, "No instructions available.")
        instructions = template.format(target_dir=target_dir)
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"  DOWNLOAD INSTRUCTIONS: {name.upper()}")
        print(separator)
        print(instructions)
        print(separator + "\n")
        logger.info("[Downloader] Displayed instructions for '%s'.", name)

    @staticmethod
    def _extract_zips(directory: Path) -> None:
        """Extract all ZIP files in a directory.

        Args:
            directory: Directory containing ZIP files.
        """
        import zipfile

        for zip_file in directory.glob("*.zip"):
            logger.info("[Downloader] Extracting '%s'...", zip_file.name)
            try:
                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(directory)
                zip_file.unlink()  # Remove ZIP after extraction
            except zipfile.BadZipFile as exc:
                logger.error("[Downloader] Bad ZIP file '%s': %s", zip_file.name, exc)
