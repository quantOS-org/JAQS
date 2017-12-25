"""
Classes defined in analyze module help automate analysis of trading results.

It takes a CSV file, trades.csv, which contains trading records, and a JSON file,
configs.json, which contains some necessary configurations. Then it can automatically
calculate PnL, position and various trade statistics and generate an HTML report.

Usage:
    ta.initialize(dataview=dv, file_folder=backtest_result_dir_path)
    ta.do_analyze(result_dir=backtest_result_dir_path, selected_sec=list(ta.universe)[:3], brinson_group='sw1')
"""

from .analyze import EventAnalyzer, AlphaAnalyzer


__all__ = ['EventAnalyzer', 'AlphaAnalyzer']