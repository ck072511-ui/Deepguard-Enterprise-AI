"""
DeepGuard — frontend/client.py

Enterprise-grade HTTP API client with retry logic, connection pooling,
timeout handling, offline detection, and automatic reconnection.
"""

import logging
import os
import time
from typing import Any, Callable, TypeVar
from functools import wraps
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

T = TypeVar('T')


def resolve_api_base_url(base_url: str | None = None) -> str:
    """Resolve the backend API base URL using st.secrets, environment variables, or defaults."""
    if base_url:
        return base_url.rstrip("/")

    resolved = None

    # 1. Try Streamlit Secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            resolved = st.secrets.get("DEEPGUARD_API_URL")
            if not resolved and "general" in st.secrets:
                resolved = st.secrets["general"].get("DEEPGUARD_API_URL")
    except Exception as e:
        logger.debug(f"Streamlit secrets not available or failed to load: {e}")

    # 2. Try Environment Variables
    if not resolved:
        env_candidates = [
            os.getenv("DEEPGUARD_API_URL"),
            os.getenv("DEEPGUARD_API_BASE_URL"),
            os.getenv("BACKEND_URL"),
            os.getenv("API_BASE_URL"),
        ]
        for candidate in env_candidates:
            if candidate:
                resolved = candidate
                break

    # 3. Fallback to Production Render URL
    if not resolved:
        resolved = "https://deepguard-enterprise-ai.onrender.com/api/v1"

    resolved = resolved.rstrip("/")
    
    # Print to Streamlit stdout/stderr logs and python logger
    print(f"[DeepGuard ST Startup] Resolved DEEPGUARD_API_URL: {resolved}")
    logger.info(f"Resolved DEEPGUARD_API_URL: {resolved}")
    
    return resolved


class APIException(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: int | None = None, response_data: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class ConnectionError(APIException):
    """Raised when unable to connect to backend."""
    pass


class TimeoutError(APIException):
    """Raised when request times out."""
    pass


class AuthenticationError(APIException):
    """Raised when authentication fails."""
    pass


class ServerError(APIException):
    """Raised when server returns 5xx error."""
    pass


class ClientError(APIException):
    """Raised when client sends invalid request (4xx)."""
    pass


def with_retry(max_retries: int = 3, backoff_factor: float = 0.5):
    """
    Decorator to retry failed API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff delay
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, ServerError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed.")
                except (ClientError, AuthenticationError) as e:
                    # Don't retry client errors or auth failures
                    raise e
            
            # If we exhaust retries, raise the last exception
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


class DeepGuardAPIClient:
    """
    Enterprise-grade HTTP API client for DeepGuard backend.
    
    Features:
    - Connection pooling with persistent sessions
    - Automatic retry with exponential backoff
    - Timeout handling (configurable per endpoint)
    - Offline detection and circuit breaker
    - Authentication token management
    - Comprehensive error handling and logging
    """

    def __init__(
        self, 
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        pool_connections: int = 10,
        pool_maxsize: int = 20,
    ) -> None:
        """
        Initialize API client with enterprise configurations.
        
        Args:
            base_url: Backend API base URL (default: from env or localhost:8000)
            api_key: API authentication key (default: from env)
            timeout: Default request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            pool_connections: Number of connection pools
            pool_maxsize: Max connections per pool
        """
        self.base_url = resolve_api_base_url(base_url)
        self.api_key = api_key or os.getenv("DEEPGUARD_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Circuit breaker state
        self._is_online = True
        self._last_health_check = datetime.min
        self._health_check_interval = timedelta(seconds=30)
        self._consecutive_failures = 0
        self._circuit_breaker_threshold = 5
        
        # Initialize session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": "DeepGuard-Streamlit/1.0",
            "Accept": "application/json",
        })
        
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}"
            })
        
        logger.info(f"API Client initialized: {self.base_url}")

    def is_online(self) -> bool:
        """
        Check if backend is reachable (with caching to avoid overhead).
        
        Returns:
            True if backend is healthy, False otherwise
        """
        now = datetime.now()
        
        # Use cached result if recent
        if now - self._last_health_check < self._health_check_interval:
            return self._is_online
        
        # Perform actual health check
        self._last_health_check = now
        try:
            health = self.get_health()
            self._is_online = health is not None and health.get("status") in ["healthy", "ok"]
            if self._is_online:
                self._consecutive_failures = 0
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._circuit_breaker_threshold:
                self._is_online = False
                logger.error("Circuit breaker tripped: backend marked offline")
        
        return self._is_online

    def _handle_response(self, response: requests.Response) -> dict[str, Any]:
        """
        Handle API response and raise appropriate exceptions.
        
        Args:
            response: Response object from requests
            
        Returns:
            Parsed JSON response data
            
        Raises:
            AuthenticationError: 401 Unauthorized
            ClientError: 4xx client errors
            ServerError: 5xx server errors
        """
        try:
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.HTTPError as e:
            status_code = response.status_code
            
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", response.text)
            except Exception:
                error_msg = response.text
            
            if status_code == 401:
                raise AuthenticationError(
                    f"Authentication failed: {error_msg}",
                    status_code=status_code,
                    response_data=error_data if 'error_data' in locals() else None
                )
            elif 400 <= status_code < 500:
                raise ClientError(
                    f"Client error {status_code}: {error_msg}",
                    status_code=status_code,
                    response_data=error_data if 'error_data' in locals() else None
                )
            elif 500 <= status_code < 600:
                raise ServerError(
                    f"Server error {status_code}: {error_msg}",
                    status_code=status_code,
                    response_data=error_data if 'error_data' in locals() else None
                )
            else:
                raise APIException(
                    f"HTTP error {status_code}: {error_msg}",
                    status_code=status_code,
                    response_data=error_data if 'error_data' in locals() else None
                )
        except requests.Timeout as e:
            raise TimeoutError(f"Request timed out: {str(e)}")
        except requests.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to backend: {str(e)}")

    def get_health(self) -> dict[str, Any] | None:
        """
        Call GET /health to fetch service health status.
        
        Returns:
            Health status dict or None if unreachable
        """
        try:
            endpoint = urljoin(self.base_url.rstrip("/") + "/", "health")
            response = self.session.get(endpoint, timeout=60)
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return None

    @with_retry(max_retries=2)
    def get_history_stats(self) -> dict[str, Any]:
        """
        Call GET /history/stats to fetch aggregated statistics.
        
        Returns:
            Stats dict with total, real, fake, avg_confidence, etc.
            
        Raises:
            APIException: On any API error
        """
        response = self.session.get(
            urljoin(self.base_url.rstrip("/") + "/", "history/stats"),
            timeout=self.timeout
        )
        return self._handle_response(response)

    @with_retry(max_retries=2)
    def get_history(
        self,
        page: int = 1,
        page_size: int = 20,
        media_type: str | None = None,
        label: int | None = None,
        status: str | None = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> dict[str, Any]:
        """
        Call GET /history to fetch paginated detection records.
        
        Args:
            page: Page number (1-indexed)
            page_size: Results per page
            media_type: Filter by 'image' or 'video'
            label: Filter by label (0=REAL, 1=FAKE)
            status: Filter by 'processing', 'completed', 'failed'
            sort_by: Sort field ('created_at', 'completed_at', 'confidence')
            order: Sort order ('asc', 'desc')
            
        Returns:
            Paginated response with items, total, page info
            
        Raises:
            APIException: On any API error
        """
        params = {
            "page": page,
            "page_size": page_size,
            "sort_by": sort_by,
            "order": order,
        }
        
        if media_type:
            params["media_type"] = media_type
        if label is not None:
            params["label"] = label
        if status:
            params["status"] = status
        
        response = self.session.get(
            urljoin(self.base_url.rstrip("/") + "/", "history"),
            params=params,
            timeout=self.timeout
        )
        return self._handle_response(response)

    @with_retry(max_retries=2)
    def list_models(self) -> list[dict[str, Any]]:
        """
        Call GET /models to fetch all model entries.
        
        Returns:
            List of model dicts
            
        Raises:
            APIException: On any API error
        """
        response = self.session.get(
            urljoin(self.base_url.rstrip("/") + "/", "models"),
            timeout=self.timeout
        )
        return self._handle_response(response)

    @with_retry(max_retries=2)
    def register_model(self, name: str, version: str, registry_path: str) -> dict[str, Any]:
        """
        Call POST /models to add a new model version.
        
        Args:
            name: Model name
            version: Version string
            registry_path: Path to model weights
            
        Returns:
            Created model dict
            
        Raises:
            APIException: On any API error
        """
        payload = {
            "name": name,
            "version": version,
            "registry_path": registry_path
        }
        response = self.session.post(
            urljoin(self.base_url.rstrip("/") + "/", "models"),
            json=payload,
            timeout=self.timeout
        )
        return self._handle_response(response)

    @with_retry(max_retries=2)
    def activate_model(self, model_id: str) -> dict[str, Any]:
        """
        Call POST /models/{id}/activate to activate a model.
        
        Args:
            model_id: Model ID to activate
            
        Returns:
            Activation response dict
            
        Raises:
            APIException: On any API error
        """
        response = self.session.post(
            urljoin(self.base_url.rstrip("/") + "/", f"models/{model_id}/activate"),
            timeout=self.timeout
        )
        return self._handle_response(response)

    @with_retry(max_retries=1)  # Fewer retries for heavy uploads
    def detect_image(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """
        Call POST /detect to perform deepfake detection on image.
        
        Args:
            file_bytes: Image file bytes
            filename: Original filename
            
        Returns:
            Detection result dict with label, confidence, etc.
            
        Raises:
            APIException: On any API error
        """
        files = {"file": (filename, file_bytes, "image/jpeg")}
        response = self.session.post(
            urljoin(self.base_url.rstrip("/") + "/", "detect"),
            files=files,
            timeout=60  # Extended timeout for image processing
        )
        return self._handle_response(response)

    @with_retry(max_retries=1)  # Fewer retries for heavy uploads
    def detect_video(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """
        Call POST /detect to perform deepfake detection on video.
        
        Args:
            file_bytes: Video file bytes
            filename: Original filename
            
        Returns:
            Detection result dict with label, confidence, etc.
            
        Raises:
            APIException: On any API error
        """
        files = {"file": (filename, file_bytes, "video/mp4")}
        response = self.session.post(
            urljoin(self.base_url.rstrip("/") + "/", "detect"),
            files=files,
            timeout=300  # Extended timeout for video processing
        )
        return self._handle_response(response)

    def close(self) -> None:
        """Close the session and release resources."""
        self.session.close()
        logger.info("API client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure session is closed."""
        self.close()

