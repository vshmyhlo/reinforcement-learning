from envs.memory_test import SeqCopy


def test_copy_good():
    env = SeqCopy(seq_size=2)
    seq = []

    o = env.reset()
    seq.append(o)
    assert o != 0

    o, r, d, _ = env.step(0)
    seq.append(o)
    assert o != 0
    assert r == 0
    assert not d

    o, r, d, _ = env.step(0)
    assert o == 0
    assert r == 0
    assert not d

    o, r, d, _ = env.step(seq[0])
    assert o == 0
    assert r == 1
    assert not d

    o, r, d, _ = env.step(seq[1])
    assert o == 0
    assert r == 1
    assert d


def test_copy_bad():
    env = SeqCopy(seq_size=2)
    seq = []

    o = env.reset()
    seq.append(o)
    assert o != 0

    o, r, d, _ = env.step(1)
    seq.append(o)
    assert o != 0
    assert r == -1
    assert not d

    o, r, d, _ = env.step(1)
    assert o == 0
    assert r == -1
    assert not d

    o, r, d, _ = env.step(0)
    assert o == 0
    assert r == 0
    assert not d

    o, r, d, _ = env.step(0)
    assert o == 0
    assert r == 0
    assert d
