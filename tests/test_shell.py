from ppqm.utils import shell


def test_steam_timeout() -> None:

    cmd = "echo 'hello_world' && sleep 1 && sleep 1 && sleep 1 && echo 'too_slow'"
    timeout = 1  # seconds

    stdout, _ = shell.execute(cmd, timeout=timeout)

    assert stdout is None
