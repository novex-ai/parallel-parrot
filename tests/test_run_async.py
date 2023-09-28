import parallel_parrot as pp


def test_run_async():
    async def async_fn():
        return 1

    assert pp.run_async(async_fn()) == 1
