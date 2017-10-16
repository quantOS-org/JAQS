-[] Order Matching:
    LIMIT ORDER mathing for event-driven; generate trades for alpha strategy.
-[] Extract PortfolioManager from a member of Strategy
-[x] Adjust time of re-balance, send orders and mathcing:
     plan after close, send orders after open
-[] Record necessary information during backtest
-[] improve code: ongoine
-[x] conversion between trade results and dict/list
-[x] Separate PnL analysis module, can be combined with DataRecorder
     backtest -> trades & configs -> analysis
-[] Resolution of fill price of stocks in China is 0.01
-[] Calendar Class

# single factor test:
-[] add industry neutral option
-[x] automatically expand data of low frequency when OP2 encountere
    1. Binary Operators (+ - * /): isinstance(x, df) and isinstance(y, df) is df and x.freq != y.freq
    2. Cross Section Functions (Max, Rank): must expand to daily
-[x] how to store quarterly data in dataview object
-[x] provide a simple API to analyze formula
-[x] relative return to benchmark
-[x] return array of data
-[x] adjust mode: default post
-[] factor = close - close of benchmark: how?

- modification of financial statement data
    for 1 security, 1 report_date, multiple ann_date exist because of modification.
    This can be separated by their statement type.
    For each modification, we modify the raw field value, then calculate formula again, finally modify results after new ann_date.

    After modification, original entry will be deleted, then, two new entries with different report_type will be added,
    one is the same with the original, the other is new.

-[] when should we add trade_date, ann_date, report_date fields

# DataView
-[] when fetching data, cache fetched data. So if fail, we do not need to fetch all data again.
-[] if data of some symbols is missing, dv.data_d or dv.data_q will be wrong
-[] '&&' operator can not be True in isOps2()

# Code Improvement of DataView
-[] improve get, get_quarter_ts methods.

# Latest Plan
1. single factor pre-process: extreme, standardize, neutral
2. study alpha signal research (IC distribution on time series and securities) and factor -> weights -> portfolio return
3. multi-factor combination
4. backtest analysis: portfolio return attribution -- at each time, contribution of each return.

-[] API rename

# Data
- self defined index; commidity index

# Backtest
-[] do not trade symbols that are not index members

# Analyze
-[] jinja2 search path does not work on Windows
