# encoding: utf-8
from jaqs.data.basic.instrument import InstManager


def test_inst_manager():
    mgr = InstManager()
    mgr.load_instruments()
    sym = '000001.SZ'
    inst_obj = mgr.get_intruments(sym)
    assert inst_obj.market == 'SZ'
    assert inst_obj.symbol == sym
    assert inst_obj.multiplier == 1
    assert inst_obj.inst_type == '1'


if __name__ == "__main__":
    test_inst_manager()