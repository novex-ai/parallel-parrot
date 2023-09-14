import asyncio
from typing import Coroutine, Any

try:
    import uvloop  # type: ignore
except ImportError:
    uvloop_installed = False
else:
    uvloop_installed = True

import nest_asyncio  # type: ignore

from .util import logger


def run_async(coro: Coroutine) -> Any:
    """
    Run an async coroutine synchronously.
    Try to use uvloop for faster i/o if possible
    Fall back to monkey-patching the current event loop if necessary
    """
    if uvloop_installed and _safe_get_running_loop() is None:
        pre_existing_policy = asyncio.get_event_loop_policy()
        logger.debug("using uvloop for faster asyncio")
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    else:
        pre_existing_policy = None
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            logger.warn(
                "monkey-patching the currently running event loop using nest_asyncio."
            )
            nest_asyncio.apply()
            return asyncio.run(coro)
        else:
            raise
    finally:
        if pre_existing_policy:
            # clean up after ourselves, to avoid creating problems in code executed after this
            asyncio.set_event_loop_policy(pre_existing_policy)


def _safe_get_running_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
