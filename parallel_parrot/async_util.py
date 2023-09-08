import asyncio
from typing import Coroutine, Any
import sys

try:
    import uvloop  # type: ignore
except ImportError:
    uvloop_installed = False
else:
    uvloop_installed = True


def sync_run(runnable: Coroutine) -> Any:
    """
    Run an async function synchronously.
    """
    if not uvloop_installed:
        # fall back to asyncio.run if uvloop is not supported
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
