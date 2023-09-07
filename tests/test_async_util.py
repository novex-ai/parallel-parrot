from parallel_parrot.async_util import sync_run


def test_sync_run():
    async def async_fn():
        return 1

    assert sync_run(async_fn()) == 1
