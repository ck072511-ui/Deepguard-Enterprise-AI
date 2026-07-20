"""
DeepGuard — frontend/utils/notifications.py

Toast notifications and error handling utilities for Streamlit UI.
"""

import streamlit as st
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')


class NotificationManager:
    """Centralized notification system for user feedback."""
    
    @staticmethod
    def success(message: str, icon: str = "✅") -> None:
        """Display success notification."""
        st.success(f"{icon} {message}", icon="✅")
    
    @staticmethod
    def error(message: str, icon: str = "❌") -> None:
        """Display error notification."""
        st.error(f"{icon} {message}", icon="🚨")
    
    @staticmethod
    def warning(message: str, icon: str = "⚠️") -> None:
        """Display warning notification."""
        st.warning(f"{icon} {message}", icon="⚠️")
    
    @staticmethod
    def info(message: str, icon: str = "ℹ️") -> None:
        """Display info notification."""
        st.info(f"{icon} {message}", icon="ℹ️")
    
    @staticmethod
    def loading(message: str) -> Any:
        """Display loading spinner context manager."""
        return st.spinner(f"⏳ {message}")


def handle_api_errors(fallback_message: str = "An error occurred"):
    """
    Decorator to handle API errors gracefully and show user-friendly notifications.
    
    Args:
        fallback_message: Message to show if error handling fails
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T | None:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                
                # Import here to avoid circular dependency
                from frontend.client import (
                    ConnectionError, 
                    TimeoutError, 
                    AuthenticationError,
                    ServerError,
                    ClientError,
                    APIException
                )
                
                if isinstance(e, ConnectionError):
                    NotificationManager.error(
                        "Cannot connect to backend server. Please check if the API is running."
                    )
                elif isinstance(e, TimeoutError):
                    NotificationManager.error(
                        "Request timed out. The server is taking too long to respond."
                    )
                elif isinstance(e, AuthenticationError):
                    NotificationManager.error(
                        "Authentication failed. Please check your API key in settings."
                    )
                elif isinstance(e, ServerError):
                    NotificationManager.error(
                        f"Server error: {error_msg}. Please try again later."
                    )
                elif isinstance(e, ClientError):
                    NotificationManager.warning(
                        f"Invalid request: {error_msg}"
                    )
                elif isinstance(e, APIException):
                    NotificationManager.error(
                        f"API error: {error_msg}"
                    )
                else:
                    NotificationManager.error(
                        f"{fallback_message}: {error_msg}"
                    )
                
                return None
        
        return wrapper
    return decorator


def with_loading(message: str = "Processing..."):
    """
    Decorator to show loading spinner during function execution.
    
    Args:
        message: Loading message to display
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with NotificationManager.loading(message):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global notification manager instance
notify = NotificationManager()
