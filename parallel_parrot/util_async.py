import asyncio
from typing import Coroutine, Any
import sys
import warnings

try:
    import uvloop  # type: ignore
except ImportError:
    uvloop_installed = False
else:
    uvloop_installed = True


from .util import logger


def is_inside_event_loop() -> bool:
    """
    Test if we are within an event loop, as with ipython autoawait.
    This means that we should use await directly.
    https://ipython.readthedocs.io/en/stable/interactive/autoawait.html
    """
    warnings.filterwarnings(
        "ignore",
        message="coroutine 'sleep' was never awaited",
        module="parallel_parrot",
    )
    try:
        asyncio.run(asyncio.sleep(0))
        return False
    except RuntimeError:
        return True


def register_uvloop() -> None:
    """
    Register uvloop as the asyncio event loop.
    uvloop is a faster implementation of the asyncio event loop.
    """
    if uvloop_installed:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("changed the asyncio event loop policy to uvloop")
    else:
        logger.warn("tried to register uvloop but it is not installed")


def sync_run(runnable: Coroutine) -> Any:
    """
    Run an async coroutine synchronously.
    """
    if is_inside_event_loop():
        raise Exception(
            "Do not use sync_run() and instead just use await."
            " This is because IPython autoawait is enabled."
            " see https://ipython.readthedocs.io/en/stable/interactive/autoawait.html"
        )
    if not uvloop_installed:
        # fall back to asyncio.run if uvloop is not supported, as on windows
        return asyncio.run(runnable)
    if sys.version_info >= (3, 11):
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            return runner.run(runnable)
    else:
        loop = uvloop.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(runnable)
        finally:
            loop.close()
