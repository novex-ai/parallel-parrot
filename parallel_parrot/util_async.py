import asyncio
from typing import Coroutine, Any
import sys

try:
    import uvloop  # type: ignore
except ImportError:
    uvloop_installed = False
else:
    uvloop_installed = True


from .util import logger


def is_ipython_autoawait() -> bool:
    """
    Test if IPython autoawait is enabled.
    This allows us to test if ipython is altering our python repl, making sync_run() fail.
    https://ipython.readthedocs.io/en/stable/interactive/autoawait.html
    """
    try:
        if "get_ipython" in globals():
            ipython = globals().get("get_ipython")()  # type: ignore
            if ipython and ipython.autoawait:
                return True
    except Exception:
        logger.warn("Failed to check if IPython autoawait is enabled.", exc_info=True)
    return False


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
    if is_ipython_autoawait():
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
