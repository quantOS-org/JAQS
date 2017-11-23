# encoding: utf-8
from jaqs.util.sequence import SequenceGenerator


def test_seq_gen():
    sg = SequenceGenerator()
    for i in range(1, 999):
        assert sg.get_next('order') == i

    text = 'trade'
    sg.get_next(text)
    sg.get_next(text)
    for i in range(3, 999):
        assert sg.get_next(text) == i


if __name__ == "__main__":
    test_seq_gen()
