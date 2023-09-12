from parallel_parrot.util_async import is_ipython_autoawait, register_uvloop, sync_run


def test_is_ipython_autoawait():
    assert is_ipython_autoawait() is False


def test_register_uvloop():
    register_uvloop()


def test_sync_run():
    async def async_fn():
        return 1

    assert sync_run(async_fn()) == 1
