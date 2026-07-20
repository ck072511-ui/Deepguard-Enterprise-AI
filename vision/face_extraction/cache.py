"""
DeepGuard — vision/face_extraction/cache.py

Thread-safe face crop cache with two storage tiers:

  1. In-memory LRU cache (fast, bounded, process-lifetime)
  2. Disk cache (persistent, survives restarts, uses joblib/pickle)

Cache keys are derived from a BLAKE2b hash of the raw image bytes plus
a config fingerprint, making them content-addressable and collision-proof.

Thread safety is guaranteed via a ``threading.Lock`` around all dict ops.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _image_key(image: np.ndarray, config_fingerprint: str = "") -> str:
    """Derive a deterministic cache key from image content and config.

    Uses BLAKE2b (faster than SHA-256 for small buffers, same security).

    Args:
        image:              RGB numpy array whose bytes form the hash input.
        config_fingerprint: Short hex string identifying the detection config.

    Returns:
        32-character hexadecimal key string.
    """
    raw = image.tobytes() + config_fingerprint.encode()
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


class LRUCache:
    """Thread-safe LRU (Least Recently Used) in-memory cache.

    Stores arbitrary values indexed by string keys. Evicts the
    least-recently-used entry when ``max_size`` is exceeded.

    Args:
        max_size: Maximum number of entries to retain in memory.
    """

    def __init__(self, max_size: int = 1024) -> None:
        self._max_size = max(1, max_size)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Retrieve a value by key, or None if not cached.

        Args:
            key: Cache key string.

        Returns:
            Cached value or None.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Insert or update a cache entry.

        Evicts the LRU entry if at capacity.

        Args:
            key:   Cache key string.
            value: Value to store.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a single entry from the cache.

        Args:
            key: Cache key to remove (no-op if not present).
        """
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Return the current number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate in [0.0, 1.0].

        Returns:
            Fraction of requests that were cache hits. 0.0 if no requests.
        """
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics dictionary.

        Returns:
            Dict with keys: size, max_size, hits, misses, hit_rate.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }


class FaceCache:
    """Two-tier face crop cache: LRU memory + optional disk persistence.

    Caches extracted face crops (list of numpy arrays) keyed by image
    content hash + config fingerprint.

    Args:
        memory_max_size: Maximum in-memory LRU entries.
        disk_cache_dir:  Optional directory for disk-backed cache.
                         If None, disk caching is disabled.
        config_fingerprint: Short string identifying the extraction config
                            (included in cache keys to invalidate stale entries).
    """

    def __init__(
        self,
        memory_max_size: int = 2048,
        disk_cache_dir: Path | str | None = None,
        config_fingerprint: str = "",
    ) -> None:
        self._memory = LRUCache(max_size=memory_max_size)
        self._config_fp = config_fingerprint
        self._disk_dir: Path | None = None

        if disk_cache_dir is not None:
            self._disk_dir = Path(disk_cache_dir)
            self._disk_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("FaceCache: disk cache enabled at '%s'.", self._disk_dir)

    def get(self, image: np.ndarray) -> list[np.ndarray] | None:
        """Look up cached face crops for an image.

        Checks the memory cache first, then the disk cache.
        On a disk hit, promotes the result to memory.

        Args:
            image: RGB image to look up (hashed by content).

        Returns:
            List of face crop arrays, or None if not cached.
        """
        key = _image_key(image, self._config_fp)

        # 1. Memory tier
        result = self._memory.get(key)
        if result is not None:
            return result  # type: ignore[return-value]

        # 2. Disk tier
        if self._disk_dir is not None:
            disk_path = self._disk_dir / f"{key}.pkl"
            if disk_path.exists():
                try:
                    with disk_path.open("rb") as f:
                        faces = pickle.load(f)
                    self._memory.put(key, faces)  # promote to memory
                    logger.debug("FaceCache disk hit: key=%s", key[:8])
                    return faces  # type: ignore[return-value]
                except Exception as exc:
                    logger.warning("FaceCache disk read failed (%s): %s", key[:8], exc)
                    disk_path.unlink(missing_ok=True)

        return None

    def put(self, image: np.ndarray, faces: list[np.ndarray]) -> None:
        """Store extracted face crops for an image.

        Writes to memory immediately. Writes to disk asynchronously if
        disk caching is enabled.

        Args:
            image: Source RGB image (used to compute cache key).
            faces: List of extracted face crop arrays to store.
        """
        key = _image_key(image, self._config_fp)
        self._memory.put(key, faces)

        if self._disk_dir is not None:
            disk_path = self._disk_dir / f"{key}.pkl"
            if not disk_path.exists():
                try:
                    with disk_path.open("wb") as f:
                        pickle.dump(faces, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.debug("FaceCache disk write: key=%s", key[:8])
                except Exception as exc:
                    logger.warning("FaceCache disk write failed (%s): %s", key[:8], exc)

    def invalidate(self, image: np.ndarray) -> None:
        """Remove cached entry for a specific image.

        Args:
            image: Source image whose cache entry should be deleted.
        """
        key = _image_key(image, self._config_fp)
        self._memory.invalidate(key)
        if self._disk_dir is not None:
            disk_path = self._disk_dir / f"{key}.pkl"
            disk_path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Clear all in-memory cache entries. Disk entries are preserved."""
        self._memory.clear()

    def clear_disk(self) -> int:
        """Delete all files from the disk cache directory.

        Returns:
            Number of files deleted.
        """
        if self._disk_dir is None:
            return 0
        deleted = 0
        for pkl_file in self._disk_dir.glob("*.pkl"):
            try:
                pkl_file.unlink()
                deleted += 1
            except OSError:
                pass
        logger.info("FaceCache: cleared %d disk entries.", deleted)
        return deleted

    def stats(self) -> dict[str, Any]:
        """Return combined cache statistics.

        Returns:
            Dict with memory stats and disk file count.
        """
        disk_count = 0
        if self._disk_dir is not None and self._disk_dir.exists():
            disk_count = sum(1 for _ in self._disk_dir.glob("*.pkl"))
        stats = self._memory.stats()
        stats["disk_entries"] = disk_count
        stats["disk_enabled"] = self._disk_dir is not None
        return stats
