import ppqm


def test_steam_timeout():

    cmd = "echo 'hello_world' && sleep 1 && sleep 1 && sleep 1 && echo 'too_slow'"
    timeout = 1  # seconds

    stdout, stderr = ppqm.shell.execute(cmd, timeout=timeout)

    assert stdout is None
