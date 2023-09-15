from typing import Any, Coroutine

from asyncio_anywhere import asyncio_run


def run_async(coro: Coroutine) -> Any:
    return asyncio_run(coro)
