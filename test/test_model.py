# encoding: utf-8

from __future__ import absolute_import, division, print_function

from jaqs.trade import model


def test_model():
    model.StockSelector()
    model.SimpleCostModel()
    model.AlphaContext()
    model.FactorRiskModel()
    model.FactorSignalModel()
    