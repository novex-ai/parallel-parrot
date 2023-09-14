from parallel_parrot.util_async import run_async


def test_run_async():
    async def async_fn():
        return 1

    assert run_async(async_fn()) == 1
