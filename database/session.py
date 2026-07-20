"""
DeepGuard — database/session.py

Database engine setup and async session factory.
Loads settings from configs/database_config.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import AsyncGenerator
import yaml
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# 1. Resolve database URL from environment or configuration file
db_url = os.getenv("DATABASE_URL")
echo = False

if not db_url:
    # Fallback to local config file
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "database_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        db_url = cfg.get("database", {}).get("url", "sqlite+aiosqlite:///./database/deepguard.db")
        echo = cfg.get("database", {}).get("echo", False)
    else:
        db_url = "sqlite+aiosqlite:///./database/deepguard.db"

# 2. Build connection args based on dialect
connect_args = {}
if "sqlite" in db_url:
    connect_args["check_same_thread"] = False

# 3. Create Async Engine
engine = create_async_engine(
    db_url,
    echo=echo,
    connect_args=connect_args,
)

# 4. Create Async Session Maker
async_session_factory = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency generator yielding active async database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
