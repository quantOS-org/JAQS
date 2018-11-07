# encoding: utf-8
"""


"""
from __future__ import print_function
import os

try:
    basestring
except NameError:
    basestring = str

import numpy as np
import pandas as pd
import re

import jaqs.util as jutil
from jaqs.data.align import align
from jaqs.data.py_expression_eval import Parser


class FactorDef:
    def __init__(self, name, args, body, is_quarterly=False):
        self.name = name
        self.body = body
        self.args = args
        self.is_quarterly = is_quarterly


class FactorFunc:
    def __init__(self, dv, factor, allow_future_data):
        self._factor = factor
        self._dv = dv
        self._allow_future_data = allow_future_data

    def __call__(self, *args, **kwargs):

        parser = self._dv._create_parser(allow_future_data=self._allow_future_data)

        # print("exec factor: " + self._factor.name + "(" + ','.join(self._factor.args) + ")=" + self._factor.body)
        expr = parser.parse(self._factor.body)

        var_df_dic = dict()
        var_list = expr.variables()

        for var in var_list:
            if var in self._factor.args:
                i = self._factor.args.index(var)
                var_df_dic[var] = args[i]
            elif var in parser.functions:
                if var in self._dv._import_factors and not self._dv._import_factors[var].args:
                    var_df_dic[var] = self._dv._get_var(var, self._allow_future_data)
            else:
                var_df_dic[var] = self._dv._get_var(var, self._allow_future_data)

        # TODO: send ann_date into expr.evaluate. We assume that ann_date of all fields of a symbol is the same
        df_ann = self._dv._get_ann_df()

        df_eval = parser.evaluate(var_df_dic, ann_dts=df_ann, trade_dts=self._dv.dates)

        return df_eval



class ResReturnFunc:
    """
    ResReturn(close, m, n)
    """
    def __init__(self, dv, allow_future_data):
        self._dv = dv
        self._allow_future_data = allow_future_data

    def __call__(self, *args, **kwargs):

        parser = self._dv._create_parser(allow_future_data=self._allow_future_data)

        # print("exec factor: " + self._factor.name + "(" + ','.join(self._factor.args) + ")=" + self._factor.body)
        field_name = args[0].columns.name
        formula = "Return({0}, {1}, {2}) - Return({3},{1},{2})".format(
            field_name, args[1], args[2],
            "bm_" + field_name
        )
        expr = parser.parse(formula)

        var_df_dic = dict()
        var_list = expr.variables()

        for var in var_list:
            # if var in self._factor.args:
            #     i = self._factor.args.index(var)
            #     var_df_dic[var] = args[i]
            if var in parser.functions:
                if var in self._dv._import_factors and not self._dv._import_factors[var].args:
                    var_df_dic[var] = self._dv._get_var(var, self._allow_future_data)
            else:
                var_df_dic[var] = self._dv._get_var(var, self._allow_future_data)

        # TODO: send ann_date into expr.evaluate. We assume that ann_date of all fields of a symbol is the same
        df_ann = self._dv._get_ann_df()

        df_eval = parser.evaluate(var_df_dic, ann_dts=df_ann, trade_dts=self._dv.dates)

        return df_eval


# class LabelDef(FactorDef):
#     pass
#
#
# class LabelFunc(FactorFunc):
#     pass


class DataView(object):
    """
    Prepare data before research / trade. Support file I/O.
    Support: add field, add formula, save / load.

    Attributes
    ----------
    data_api : RemoteDataService
    symbol : list
    start_date : int
    end_date : int
    fields : list
    freq : int
    market_daily_fields, reference_daily_fields : list
    data_d : pd.DataFrame
        All daily frequency data will be merged and stored here.
        index is date, columns is symbol-field MultiIndex
    data_q : pd.DataFrame
        All quarterly frequency data will be merged and stored here.
        index is date, columns is symbol-field MultiIndex

    """

    def __init__(self):
        self.data_api = None
        self._factor_df = pd.DataFrame()
        self._import_factors = {}
        self.universe = ""
        self.symbol = []
        self.benchmark = ""
        self.start_date = 0
        self.extended_start_date_d = 0
        self.extended_start_date_q = 0
        self.end_date = 0
        self.fields = []
        self.freq = 1
        self.all_price = True
        self._snapshot = None
        self.factors = []
        self.load_factors = []
        self.labels = []
        self.load_labels = []
        self.index_weights = {}
        self.industry_groups = {}

        self.meta_data_list = ['start_date', 'end_date',
                               'extended_start_date_d', 'extended_start_date_q',
                               'freq', 'fields', 'symbol', 'universe', 'benchmark',
                               'custom_daily_fields', 'custom_quarterly_fields',
                               'factors', 'load_factors',
                               'labels', 'load_labels'
                               ]

        self.adjust_mode = 'post'

        self.data_d = None
        self.data_q = None
        self._data_benchmark = None
        self._data_inst = None
        # self._data_group = None

        common_list = {'symbol', 'start_date', 'end_date'}
        market_bar_list = {'open', 'high', 'low', 'close', 'volume', 'turnover', 'vwap', 'oi'}
        market_tick_list = {'volume', 'oi',
                            'askprice1', 'askprice2', 'askprice3', 'askprice4', 'askprice5',
                            'bidprice1', 'bidprice1', 'bidprice1', 'bidprice1', 'bidprice1',
                            'askvolume1', 'askvolume2', 'askvolume3', 'askvolume4', 'askvolume5',
                            'bidvolume1', 'bidvolume2', 'bidvolume3', 'bidvolume4', 'bidvolume5'}
        # fields map
        # TODO: 'freq' is not in market_daily_fields yet.
        self.market_daily_fields = \
            {'open', 'high', 'low', 'close', 'volume', 'turnover', 'vwap', 'oi', 'trade_status',
             'open_adj', 'high_adj', 'low_adj', 'close_adj', 'vwap_adj', 'index_member', 'index_weight'}
        self.group_fields = {'sw1', 'sw2', 'sw3', 'sw4', 'zz1', 'zz2'}
        self.reference_daily_fields = \
            {"total_mv", "float_mv", "pe", "pb", "pe_ttm", "pcf_ocf", "pcf_ocfttm", "pcf_ncf",
             "pcf_ncfttm", "ps", "ps_ttm", "turnover_ratio", "free_turnover_ratio", "total_share",
             "float_share", "price_div_dps", "free_share", "np_parent_comp_ttm",
             "np_parent_comp_lyr", "net_assets", "ncf_oper_ttm", "ncf_oper_lyr", "oper_rev_ttm",
             "oper_rev_lyr", "limit_status"}
        self.fin_stat_income = \
            {"symbol", "ann_date", "start_date", "end_date",
             "comp_type_code", "comp_type_code", "act_ann_date", "start_actdate",
             "end_actdate", "report_date", "start_reportdate", "start_reportdate",
             "report_type", "report_type", "currency", "total_oper_rev", "oper_rev",
             "int_income", "net_int_income", "insur_prem_unearned",
             "handling_chrg_income", "net_handling_chrg_income", "net_inc_other_ops",
             "plus_net_inc_other_bus", "prem_income", "less_ceded_out_prem",
             "chg_unearned_prem_res", "incl_reinsurance_prem_inc",
             "net_inc_sec_trading_brok_bus", "net_inc_sec_uw_bus",
             "net_inc_ec_asset_mgmt_bus", "other_bus_income", "plus_net_gain_chg_fv",
             "plus_net_invest_inc", "incl_inc_invest_assoc_jv_entp",
             "plus_net_gain_fx_trans", "tot_oper_cost", "less_oper_cost", "less_int_exp",
             "less_handling_chrg_comm_exp", "less_taxes_surcharges_ops",
             "less_selling_dist_exp", "less_gerl_admin_exp", "less_fin_exp",
             "less_impair_loss_assets", "prepay_surr", "tot_claim_exp",
             "chg_insur_cont_rsrv", "dvd_exp_insured", "reinsurance_exp", "oper_exp",
             "less_claim_recb_reinsurer", "less_ins_rsrv_recb_reinsurer",
             "less_exp_recb_reinsurer", "other_bus_cost", "oper_profit",
             "plus_non_oper_rev", "less_non_oper_exp", "il_net_loss_disp_noncur_asset",
             "tot_profit", "inc_tax", "unconfirmed_invest_loss",
             "net_profit_incl_min_int_inc", "net_profit_excl_min_int_inc",
             "minority_int_inc", "other_compreh_inc", "tot_compreh_inc",
             "tot_compreh_inc_parent_comp", "tot_compreh_inc_min_shrhldr", "ebit",
             "ebitda", "net_profit_after_ded_nr_lp", "net_profit_under_intl_acc_sta",
             "eps_basic", "eps_diluted", "insurance_expense",
             "spe_bal_oper_profit", "tot_bal_oper_profit", "spe_bal_tot_profit",
             "tot_bal_tot_profit", "spe_bal_net_profit", "tot_bal_net_profit",
             "undistributed_profit", "adjlossgain_prevyear",
             "transfer_from_surplusreserve", "transfer_from_housingimprest",
             "transfer_from_others", "distributable_profit", "withdr_legalsurplus",
             "withdr_legalpubwelfunds", "workers_welfare", "withdr_buzexpwelfare",
             "withdr_reservefund", "distributable_profit_shrhder", "prfshare_dvd_payable",
             "withdr_othersurpreserve", "comshare_dvd_payable",
             "capitalized_comstock_div"}
        self.fin_stat_balance_sheet = \
            {"monetary_cap", "tradable_assets", "notes_rcv", "acct_rcv", "other_rcv", "pre_pay",
             "dvd_rcv", "int_rcv", "inventories", "consumptive_assets", "deferred_exp",
             "noncur_assets_due_1y", "settle_rsrv", "loans_to_banks", "prem_rcv", "rcv_from_reinsurer",
             "rcv_from_ceded_insur_cont_rsrv", "red_monetary_cap_for_sale", "other_cur_assets",
             "tot_cur_assets", "fin_assets_avail_for_sale", "held_to_mty_invest", "long_term_eqy_invest",
             "invest_real_estate", "time_deposits", "other_assets", "long_term_rec", "fix_assets",
             "const_in_prog", "proj_matl", "fix_assets_disp", "productive_bio_assets",
             "oil_and_natural_gas_assets", "intang_assets", "r_and_d_costs", "goodwill",
             "long_term_deferred_exp", "deferred_tax_assets", "loans_and_adv_granted",
             "oth_non_cur_assets", "tot_non_cur_assets", "cash_deposits_central_bank",
             "asset_dep_oth_banks_fin_inst", "precious_metals", "derivative_fin_assets",
             "agency_bus_assets", "subr_rec", "rcv_ceded_unearned_prem_rsrv", "rcv_ceded_claim_rsrv",
             "rcv_ceded_life_insur_rsrv", "rcv_ceded_lt_health_insur_rsrv", "mrgn_paid",
             "insured_pledge_loan", "cap_mrgn_paid", "independent_acct_assets", "clients_cap_deposit",
             "clients_rsrv_settle", "incl_seat_fees_exchange", "rcv_invest", "tot_assets", "st_borrow",
             "borrow_central_bank", "deposit_received_ib_deposits", "loans_oth_banks", "tradable_fin_liab",
             "notes_payable", "acct_payable", "adv_from_cust", "fund_sales_fin_assets_rp",
             "handling_charges_comm_payable", "empl_ben_payable", "taxes_surcharges_payable", "int_payable",
             "dvd_payable", "other_payable", "acc_exp", "deferred_inc", "st_bonds_payable", "payable_to_reinsurer",
             "rsrv_insur_cont", "acting_trading_sec", "acting_uw_sec", "non_cur_liab_due_within_1y", "other_cur_liab",
             "tot_cur_liab", "lt_borrow", "bonds_payable", "lt_payable", "specific_item_payable", "provisions",
             "deferred_tax_liab", "deferred_inc_non_cur_liab", "other_non_cur_liab", "tot_non_cur_liab",
             "liab_dep_other_banks_inst", "derivative_fin_liab", "cust_bank_dep", "agency_bus_liab", "other_liab",
             "prem_received_adv", "deposit_received", "insured_deposit_invest", "unearned_prem_rsrv", "out_loss_rsrv",
             "life_insur_rsrv", "lt_health_insur_v", "independent_acct_liab", "incl_pledge_loan", "claims_payable",
             "dvd_payable_insured", "total_liab", "capital_stk", "capital_reser", "special_rsrv", "surplus_rsrv",
             "undistributed_profit", "less_tsy_stk", "prov_nom_risks", "cnvd_diff_foreign_curr_stat",
             "unconfirmed_invest_loss", "minority_int", "tot_shrhldr_eqy_excl_min_int", "tot_shrhldr_eqy_incl_min_int",
             "tot_liab_shrhldr_eqy", "spe_cur_assets_diff", "tot_cur_assets_diff", "spe_non_cur_assets_diff",
             "tot_non_cur_assets_diff", "spe_bal_assets_diff", "tot_bal_assets_diff", "spe_cur_liab_diff",
             "tot_cur_liab_diff", "spe_non_cur_liab_diff", "tot_non_cur_liab_diff", "spe_bal_liab_diff",
             "tot_bal_liab_diff", "spe_bal_shrhldr_eqy_diff", "tot_bal_shrhldr_eqy_diff", "spe_bal_liab_eqy_diff",
             "tot_bal_liab_eqy_diff", "lt_payroll_payable", "other_comp_income", "other_equity_tools",
             "other_equity_tools_p_shr", "lending_funds", "accounts_receivable", "st_financing_payable", "payables"}
        self.fin_stat_cash_flow = \
            {"cash_recp_sg_and_rs", "recp_tax_rends", "net_incr_dep_cob", "net_incr_loans_central_bank",
             "net_incr_fund_borr_ofi", "cash_recp_prem_orig_inco", "net_incr_insured_dep",
             "net_cash_received_reinsu_bus", "net_incr_disp_tfa", "net_incr_int_handling_chrg", "net_incr_disp_faas",
             "net_incr_loans_other_bank", "net_incr_repurch_bus_fund", "other_cash_recp_ral_oper_act",
             "stot_cash_inflows_oper_act", "cash_pay_goods_purch_serv_rec", "cash_pay_beh_empl", "pay_all_typ_tax",
             "net_incr_clients_loan_adv", "net_incr_dep_cbob", "cash_pay_claims_orig_inco", "handling_chrg_paid",
             "comm_insur_plcy_paid", "other_cash_pay_ral_oper_act", "stot_cash_outflows_oper_act",
             "net_cash_flows_oper_act", "cash_recp_disp_withdrwl_invest", "cash_recp_return_invest",
             "net_cash_recp_disp_fiolta", "net_cash_recp_disp_sobu", "other_cash_recp_ral_inv_act",
             "stot_cash_inflows_inv_act", "cash_pay_acq_const_fiolta", "cash_paid_invest", "net_cash_pay_aquis_sobu",
             "other_cash_pay_ral_inv_act", "net_incr_pledge_loan", "stot_cash_outflows_inv_act",
             "net_cash_flows_inv_act", "cash_recp_cap_contrib", "incl_cash_rec_saims", "cash_recp_borrow",
             "proc_issue_bonds", "other_cash_recp_ral_fnc_act", "stot_cash_inflows_fnc_act", "cash_prepay_amt_borr",
             "cash_pay_dist_dpcp_int_exp", "incl_dvd_profit_paid_sc_ms", "other_cash_pay_ral_fnc_act",
             "stot_cash_outflows_fnc_act", "net_cash_flows_fnc_act", "eff_fx_flu_cash", "net_incr_cash_cash_equ",
             "cash_cash_equ_beg_period", "cash_cash_equ_end_period", "net_profit", "unconfirmed_invest_loss",
             "plus_prov_depr_assets", "depr_fa_coga_dpba", "amort_intang_assets", "amort_lt_deferred_exp",
             "decr_deferred_exp", "incr_acc_exp", "loss_disp_fiolta", "loss_scr_fa", "loss_fv_chg", "fin_exp",
             "invest_loss", "decr_deferred_inc_tax_assets", "incr_deferred_inc_tax_liab", "decr_inventories",
             "decr_oper_payable", "incr_oper_payable", "others", "im_net_cash_flows_oper_act", "conv_debt_into_cap",
             "conv_corp_bonds_due_within_1y", "fa_fnc_leases", "end_bal_cash", "less_beg_bal_cash",
             "plus_end_bal_cash_equ", "less_beg_bal_cash_equ", "im_net_incr_cash_cash_equ", "free_cash_flow",
             "spe_bal_cash_inflows_oper", "tot_bal_cash_inflows_oper", "spe_bal_cash_outflows_oper",
             "tot_bal_cash_outflows_oper", "tot_bal_netcash_outflows_oper", "spe_bal_cash_inflows_inv",
             "tot_bal_cash_inflows_inv", "spe_bal_cash_outflows_inv", "tot_bal_cash_outflows_inv",
             "tot_bal_netcash_outflows_inv", "spe_bal_cash_inflows_fnc", "tot_bal_cash_inflows_fnc",
             "spe_bal_cash_outflows_fnc", "tot_bal_cash_outflows_fnc", "tot_bal_netcash_outflows_fnc",
             "spe_bal_netcash_inc", "tot_bal_netcash_inc", "spe_bal_netcash_equ_undir", "tot_bal_netcash_equ_undir",
             "spe_bal_netcash_inc_undir", "tot_bal_netcash_inc_undir"}
        self.fin_indicator = \
            {"extraordinary", "deductedprofit", "grossmargin", "operateincome", "investincome", "stmnote_finexp",
             "stm_is", "ebit_daily", "ebitda""fcff", "fcfe", "exinterestdebt_current", "exinterestdebt_noncurrent",
             "interestdebt",
             "netdebt", "tangibleasset", "workingcapital", "networkingcapital", "investcapital", "retainedearnings",
             "eps_basic_daily",  # TODO eps_basic
             "eps_diluted", "eps_diluted2", "bps", "ocfps", "grps", "orps", "surpluscapitalps", "surplusreserveps",
             "undistributedps",
             "retainedps", "cfps", "ebitps", "fcffps", "fcfeps", "netprofitmargin", "grossprofitmargin", "cogstosales",
             "expensetosales", "profittogr", "saleexpensetogr", "adminexpensetogr", "finaexpensetogr", "impairtogr_ttm",
             "gctogr", "optogr", "ebittogr", "roe", "roe_deducted", "roa2", "roa", "roic", "roe_yearly", "roa2_yearly",
             "roe_avg",
             "operateincometoebt", "investincometoebt", "nonoperateprofittoebt", "taxtoebt", "deductedprofittoprofit",
             "salescashintoor",
             "ocftoor", "ocftooperateincome", "capitalizedtoda", "debttoassets", "assetstoequity",
             "dupont_assetstoequity",
             "catoassets", "ncatoassets", "tangibleassetstoassets", "intdebttototalcap", "equitytototalcapital",
             "currentdebttodebt",
             "longdebtodebt", "current", "quick", "cashratio", "ocftoshortdebt", "debttoequity", "equitytodebt",
             "equitytointerestdebt", "tangibleassettodebt", "tangassettointdebt", "tangibleassettonetdebt", "ocftodebt",
             "ocftointerestdebt", "ocftonetdebt", "ebittointerest", "longdebttoworkingcapital", "ebitdatodebt",
             "turndays",
             "invturndays", "arturndays", "invturn", "arturn", "caturn", "faturn", "assetsturn", "roa_yearly",
             "dupont_roa",
             "s_stm_bs", "prefinexpense_opprofit", "nonopprofit", "optoebt", "noptoebt", "ocftoprofit", "cashtoliqdebt",
             "cashtoliqdebtwithinterest", "optoliqdebt", "optodebt", "roic_yearly", "tot_faturn", "profittoop",
             "qfa_operateincome",
             "qfa_investincome", "qfa_deductedprofit", "qfa_eps", "qfa_netprofitmargin", "qfa_grossprofitmargin",
             "qfa_expensetosales",
             "qfa_profittogr", "qfa_saleexpensetogr", "qfa_adminexpensetogr", "qfa_finaexpensetogr",
             "qfa_impairtogr_ttm",
             "qfa_gctogr", "qfa_optogr", "qfa_roe", "qfa_roe_deducted", "qfa_roa", "qfa_operateincometoebt",
             "qfa_investincometoebt",
             "qfa_deductedprofittoprofit", "qfa_salescashintoor", "qfa_ocftosales", "qfa_ocftoor", "yoyeps_basic",
             "yoyeps_diluted",
             "yoyocfps", "yoyop", "yoyebt", "yoynetprofit", "yoynetprofit_deducted", "yoyocf", "yoyroe", "yoybps",
             "yoyassets",
             "yoyequity", "yoy_tr", "yoy_or", "qfa_yoygr", "qfa_cgrgr", "qfa_yoysales", "qfa_cgrsales", "qfa_yoyop",
             "qfa_cgrop",
             "qfa_yoyprofit", "qfa_cgrprofit", "qfa_yoynetprofit", "qfa_cgrnetprofit", "yoy_equity", "rd_expense",
             "waa_roe"}

        self.risk_model_fields = { "expo_value", 
                            "expo_leverage",
                            "expo_growth",
                            "expo_size",
                            "expo_liquidity",
                            "expo_shortterm_momentum",
                            "expo_mediumterm_momentum",
                            "expo_volatility",
                            "expo_exchange_rate_sensitivity",
                            "expo_bshare_market",
                            "expo_coal_consumable_fuels",
                            "expo_energy_ex_coal",
                            "expo_steel",
                            "expo_chemicals",
                            "expo_construction_materials",
                            "expo_metals_mining_ex_steel",
                            "expo_paper_forest_products",
                            "expo_commercial_professional_services",
                            "expo_electrical_equipment",
                            "expo_construction_engineering",
                            "expo_transportation_noninfrastructure",
                            "expo_machinery",
                            "expo_trading_companies_distributors_conglomerates",
                            "expo_transportation_infrastructure",
                            "expo_media",
                            "expo_retailing",
                            "expo_textiles_apparel_luxury_goods",
                            "expo_automobiles",
                            "expo_household_durables",
                            "expo_autocomponents",
                            "expo_consumer_services",
                            "expo_food_products",
                            "expo_beverages_tobacco",
                            "expo_healthcare",
                            "expo_realestate",
                            "expo_financials",
                            "expo_software_services",
                            "expo_computers_peripherals",
                            "expo_communications_equipment",
                            "expo_semiconductors_electronics",
                            "expo_telecommunication_services",
                            "expo_utilities"
                        }

        self.consensus_data = [
            "est_bps_fy0","est_cfps_fy0","est_dps_fy0","est_ebit_fy0","est_eps_fy0","est_oper_revenue_fy0","est_pb_fy0","est_pe_fy0","est_peg_fy0","est_roe_fy0","net_profit_fy0","rolling_type_fy0",
            "est_bps_fy1","est_cfps_fy1","est_dps_fy1","est_ebit_fy1","est_eps_fy1","est_oper_revenue_fy1","est_pb_fy1","est_pe_fy1","est_peg_fy1","est_roe_fy1","net_profit_fy1","rolling_type_fy1",
            "est_bps_fy2","est_cfps_fy2","est_dps_fy2","est_ebit_fy2","est_eps_fy2","est_oper_revenue_fy2","est_pb_fy2","est_pe_fy2","est_peg_fy2","est_roe_fy2","net_profit_fy2","rolling_type_fy2",
            "est_bps_fy3","est_cfps_fy3","est_dps_fy3","est_ebit_fy3","est_eps_fy3","est_oper_revenue_fy3","est_pb_fy3","est_pe_fy3","est_peg_fy3","est_roe_fy3","net_profit_fy3","rolling_type_fy3",
            "est_bps_fttm","est_cfps_fttm","est_dps_fttm","est_ebit_fttm","est_eps_fttm","est_oper_revenue_fttm","est_pb_fttm","est_pe_fttm","est_peg_fttm","est_roe_fttm","net_profit_fttm","rolling_type_fttm",
            "est_bps_yoy","est_cfps_yoy","est_dps_yoy","est_ebit_yoy","est_eps_yoy","est_oper_revenue_yoy","est_pb_yoy","est_pe_yoy","est_peg_yoy","est_roe_yoy","net_profit_yoy","rolling_type_yoy",
            "est_bps_yoy2","est_cfps_yoy2","est_dps_yoy2","est_ebit_yoy2","est_eps_yoy2","est_oper_revenue_yoy2","est_pb_yoy2","est_pe_yoy2","est_peg_yoy2","est_roe_yoy2","net_profit_yoy2","rolling_type_yoy2",
            "est_bps_cagr","est_cfps_cagr","est_dps_cagr","est_ebit_cagr","est_eps_cagr","est_oper_revenue_cagr","est_pb_cagr","est_pe_cagr","est_peg_cagr","est_roe_cagr","net_profit_cagr","rolling_type_cagr" ]

        self.stk_rating_data = [
            "est_price_30d", "est_priceinstnum_30d", "wrating_avg_30d", "wrating_downgrade_30d","wrating_instnum_30d", "wrating_maintain_30d", "wrating_numofbuy_30d", "wrating_numofhold_30d","wrating_numofoutperform_30d", "wrating_numofsell_30d", "wrating_numofunderperform_30d","wrating_upgrade_30d",
            "est_price_90d", "est_priceinstnum_90d", "wrating_avg_90d", "wrating_downgrade_90d", "wrating_instnum_90d", "wrating_maintain_90d", "wrating_numofbuy_90d","wrating_numofhold_90d", "wrating_numofoutperform_90d", "wrating_numofsell_90d","wrating_numofunderperform_90d", "wrating_upgrade_90d",
            "est_price_180d", "est_priceinstnum_180d","wrating_avg_180d","wrating_downgrade_180d", "wrating_instnum_180d", "wrating_maintain_180d", "wrating_numofbuy_180d", "wrating_numofhold_180d", "wrating_numofoutperform_180d","wrating_numofsell_180d", "wrating_numofunderperform_180d", "wrating_upgrade_180d"
        ]

        self.lgt_data = {"lgt_holding", "lgt_holding_ratio"}
        self.rating_data = {"num_rating"}

        self.default_fields = {
                          '_daily_adjust_factor', '_limit', 'adjust_factor', 'close',
                          'close_adj', 'high', 'high_adj', 'index_member', 'index_weight',
                          'low', 'low_adj', 'open', 'open_adj', 'trade_status', 'vwap',
                          'vwap_adj',
                          'sw1', 'sw2', 'sw3', 'sw4', 'zz1', 'zz2' }

        self.custom_daily_fields = []
        self.custom_quarterly_fields = []

        # co nst
        self.ANN_DATE_FIELD_NAME = 'ann_date'
        self.REPORT_DATE_FIELD_NAME = 'report_date'
        self.TRADE_STATUS_FIELD_NAME = 'trade_status'
        self.TRADE_DATE_FIELD_NAME = 'trade_date'

        self.misc_data = [ 'st_flag' ]

    # --------------------------------------------------------------------------------------------------------
    # Properties
    @property
    def data_benchmark(self):
        return self._data_benchmark

    @property
    def data_inst(self):
        """

        Returns
        -------
        pd.DataFrame

        """
        return self._data_inst

    @data_benchmark.setter
    def data_benchmark(self, df_new):
        if self.data_d is not None and df_new.shape[0] != self.data_d.shape[0]:
            raise ValueError("You must provide a DataFrame with the same shape of data_benchmark.")
        self._data_benchmark = df_new

    @property
    def dates(self):
        """
        Get daily date array of the underlying data.

        Returns
        -------
        res : np.array
            dtype: int

        """
        if self.data_d is not None:
            res = self.data_d.index.values
        elif self.data_api is not None:
            res = self.data_api.query_trade_dates(self.extended_start_date_d, self.end_date)
        else:
            raise ValueError("Cannot get dates array when neither of data and data_api exists.")

        return res

    # --------------------------------------------------------------------------------------------------------
    # Fields
    def _is_quarter_field(self, field_name):
        """
        Check whether a field name is quarterly frequency.
        field_name must be pre-defined or already added.

        Parameters
        ----------
        field_name : str

        Returns
        -------
        bool

        """
        res = (field_name in self.fin_stat_balance_sheet
               or field_name in self.fin_stat_cash_flow
               or field_name in self.fin_stat_income
               or field_name in self.fin_indicator
               or field_name in self.custom_quarterly_fields)
        return res

    def _is_daily_field(self, field_name):
        """
        Check whether a field name is daily frequency.
        field_name must be pre-defined or already added.

        Parameters
        ----------
        field_name : str

        Returns
        -------
        bool

        """
        flag = (field_name in self.market_daily_fields
                or field_name in self.reference_daily_fields
                or field_name in self.custom_daily_fields
                or field_name in self.group_fields
                or field_name in self.risk_model_fields
                or field_name in self.rating_data
                or field_name in self.lgt_data
                or field_name in self.consensus_data
                or field_name in self.stk_rating_data
                or field_name in self.misc_data)
        return flag

    def _is_predefined_field(self, field_name):
        """
        Check whether a field name can be recognized.
        field_name must be pre-defined or already added.

        Parameters
        ----------
        field_name : str

        Returns
        -------
        bool

        """
        return self._is_quarter_field(field_name) or self._is_daily_field(field_name)

    def _get_fields(self, field_type, fields, complement=False, append=False):
        """
        Get list of fields that are in ref_quarterly_fields.

        Parameters
        ----------
        field_type : {'market_daily', 'ref_daily', 'income', 'balance_sheet', 'cash_flow', 'daily', 'quarterly'
        fields : list of str
        complement : bool, optional
            If True, get fields that are NOT in ref_quarterly_fields.

        Returns
        -------
        list

        """
        pool_map = {'market_daily': self.market_daily_fields,
                    'ref_daily': self.reference_daily_fields,
                    'income': self.fin_stat_income,
                    'balance_sheet': self.fin_stat_balance_sheet,
                    'cash_flow': self.fin_stat_cash_flow,
                    'fin_indicator': self.fin_indicator,
                    'group': self.group_fields,
                    'risk_model':self.risk_model_fields,
                    'lgt_data' : self.lgt_data,
                    'rating_data': self.rating_data,
                    'consensus_data': self.consensus_data,
                    'stk_rating_data': self.stk_rating_data,
                    'misc_data' : self.misc_data
                    }
        pool_map['daily'] = set.union(pool_map['market_daily'],
                                      pool_map['ref_daily'],
                                      pool_map['group'],
                                      pool_map['risk_model'],
                                      self.custom_daily_fields)
        pool_map['quarterly'] = set.union(pool_map['income'],
                                          pool_map['balance_sheet'],
                                          pool_map['cash_flow'],
                                          pool_map['fin_indicator'],
                                          self.custom_quarterly_fields)

        pool = pool_map.get(field_type, None)
        if pool is None:
            raise NotImplementedError("field_type = {:s}".format(field_type))

        s = set.intersection(set(pool), set(fields))
        if not s:
            return []

        if complement:
            s = set(fields) - s

        if field_type == 'market_daily' and self.all_price:
            # turnover will not be adjusted
            s.update({'open', 'high', 'close', 'low', 'vwap'})

        if append:
            s.add('symbol')
            if field_type == 'market_daily' or field_type == 'ref_daily' or field_type == "risk_model":
                s.add('trade_date')
                if field_type == 'market_daily':
                    s.add(self.TRADE_STATUS_FIELD_NAME)
            elif (field_type == 'income'
                  or field_type == 'balance_sheet'
                  or field_type == 'cash_flow'
                  or field_type == 'fin_indicator'):
                s.add(self.ANN_DATE_FIELD_NAME)
                s.add(self.REPORT_DATE_FIELD_NAME)

        l = list(s)
        return l

    # --------------------------------------------------------------------------------------------------------
    # Prepare data
    def init_from_config(self, props, data_api):
        """
        Initialize various attributes like start/end date, universe/symbol, fields, etc.
        If your want to parse symbol, but use a custom benchmark index, please directly assign self.data_benchmark.

        Parameters
        ----------
        props : dict
            start_date, end_date, freq, symbol, fields, etc.
        data_api : BaseDataServer

        """
        # data_api.init_from_config(props)
        self.data_api = data_api

        sep = ','

        # initialize parameters
        self.start_date = props['start_date']
        self.extended_start_date_d = jutil.shift(self.start_date, n_weeks=-8)  # query more data
        self.extended_start_date_q = jutil.shift(self.start_date, n_weeks=-80)
        self.end_date = props['end_date']
        self.all_price = props.get('all_price', True)
        self.freq = props.get('freq', 1)

        # get and filter fields
        fields = props.get('fields', [])
        if fields:
            fields = props['fields'].split(sep)
            self.fields = [field for field in fields if self._is_predefined_field(field)]
            if len(self.fields) < len(fields):
                print("Field name [{}] not valid, ignore.".format(set.difference(set(fields), set(self.fields))))

        # append additional fields
        if self.all_price:
            self.fields.extend(['open_adj', 'high_adj', 'low_adj', 'close_adj',
                                'open', 'high', 'low', 'close',
                                'vwap', 'vwap_adj'])

        # always inclde sw1 for industry analysis
        if 'sw1' not in self.fields:
            self.fields.extend(['sw1'])

        # TODO: load factors and labels from DataCore server
        # "factors"       : "vwap=vwap; mom_1_2=mom_m_n(1,);..."
        # "load_factors"  : vwap, mom_m_n,..."
        # "labels"        : "vwap=vwap; mom_1_2=mom_m_n(1,);..."
        # "load_labels"   : "vwap, mom_m_n,..."

        # self._props = props
        self.factors = props['factors'].split(';') if 'factors' in props else []
        self.load_factors = props['load_factors'].split(',') if 'load_factors' in props else []
        self.labels = props['labels'].split(';') if 'labels' in props else []
        self.load_labels = props['load_labels'].split(',') if 'load_labels' in props else []

        if self.factors or self.load_factors or self.labels or self.load_labels:
            self._init_factors()

        # initialize universe/symbol
        universe = props.get('universe', "")
        symbol = props.get('symbol', "")
        benchmark = props.get('benchmark', '')
        # if symbol and universe:
        #     raise ValueError("Please use either [symbol] or [universe].")
        # if not (symbol or universe):
        #     raise ValueError("One of [symbol] or [universe] must be provided.")
        if universe:
            univ_list = universe.split(',')
            self.universe = univ_list
            symbols_list = []
            for univ in univ_list:
                if univ.endswith('.UNI'):
                    symbols_list.extend(data_api.query_universe_member(univ, self.extended_start_date_d, self.end_date))
                else:
                    symbols_list.extend(data_api.query_index_member(univ, self.extended_start_date_d, self.end_date))

            self.symbol = sorted(list(set(symbols_list)))
        # else:
        #    self.symbol = sorted(symbol.split(sep))

        # Merge universe and symbol
        if symbol:
            tmp = self.symbol + symbol.split(sep)
            self.symbol = sorted(list(set(tmp)))

        if benchmark:
            self.benchmark = benchmark
        else:
            if self.universe:
                if len(self.universe) > 1:
                    print("More than one universe are used: {}, "
                          "use the first one ({}) as index by default. "
                          "If you want to use other benchmark, "
                          "please specify benchmark in configs.".format(repr(self.universe), self.universe[0]))
                self.benchmark = self.universe[0]

        print("Initialize config success.")

    def _init_factors(self):
        # append all factors from db
        factor_df, msg = self.data_api.query(view='jz.factorDef')
        self._factor_df = factor_df
        if self._factor_df is None:
            raise ValueError("no factor_table: " + msg)

        for index, row in self._factor_df.iterrows():
            factor_id = row['factor_id']
            factor_body = row['factor_def']
            factor_args = list(filter(None, row['factor_args'].split(',')))
            if 'factor_quarterly' in row:
                factor_quarterly = row['factor_quarterly']
            else:
                factor_quarterly = False

            self._import_factors[factor_id] = FactorDef(factor_id, factor_args, factor_body, factor_quarterly)

        factor_label_ids = list(filter(None, list(
            (set([item.split('(')[0] for item in self.load_factors]))
                .union(set([item.split('(')[0] for item in self.load_labels]))
                .union(set([item1.split('(')[0] for item1 in [item.split('=')[-1] for item in self.factors]]))
                .union(set([item1.split('(')[0] for item1 in [item.split('=')[-1] for item in self.labels]]))
        )))

        dependencies = []
        while factor_label_ids:
            tmp = self._factor_df[self._factor_df['factor_id'].isin(factor_label_ids)]['dependency']
            factor_label_ids = []
            for dep in tmp:
                for d in dep.split(','):
                    d = d.strip()
                    if not d: continue
                    if d in self._import_factors:
                        factor_label_ids.append(d)
                    else:
                        dependencies.append(d)

        for dep in set(dependencies):
            if dep not in self.fields:
                self.fields.append(dep)

    def distributed_query(self, query_func_name, symbol, start_date, end_date, limit=100000, **kwargs):
        n_symbols = len(symbol.split(','))
        dates = self.data_api.query_trade_dates(start_date, end_date)
        n_days = len(dates)

        if n_symbols * n_days > limit:
            n = limit // n_symbols

            df_list = []
            i = 0
            pos1, pos2 = n * i, n * (i + 1) - 1
            while pos2 < n_days:
                print(pos2)
                df, msg = getattr(self.data_api, query_func_name)(symbol=symbol,
                                                                  start_date=dates[pos1], end_date=dates[pos2],
                                                                  **kwargs)
                df_list.append(df)
                i += 1
                pos1, pos2 = n * i, n * (i + 1) - 1
            if pos1 < n_days:
                df, msg = getattr(self.data_api, query_func_name)(symbol=symbol,
                                                                  start_date=dates[pos1], end_date=dates[-1],
                                                                  **kwargs)
                df_list.append(df)
            df = pd.concat(df_list, axis=0)
        else:
            df, msg = getattr(self.data_api, query_func_name)(symbol,
                                                              start_date=start_date, end_date=end_date,
                                                              **kwargs)
        return df, msg

    def prepare_data(self):
        """Prepare data for the FIRST time."""
        # prepare benchmark and group
        print("Query data...")
        data_d, data_q = self._prepare_daily_quarterly(self.fields)
        self.data_d, self.data_q = data_d, data_q
        if self.data_q is not None:
            self._prepare_report_date()
        self._align_and_merge_q_into_d()

        print("Query instrument info...")
        self._prepare_inst_info()

        print("Query adj_factor...")
        self._prepare_adj_factor()

        if self.benchmark:
            print("Query benchmark...")
            self._data_benchmark = self._prepare_benchmark()
        if self.universe:
            print("Query benchmar member info...")
            self._prepare_comp_info()

        group_fields = self._get_fields('group', self.fields)
        if group_fields:
            print("Query groups (industry)...")
            self._prepare_group(group_fields)

        self._process_data()

        print("Data has been successfully prepared.")

    @staticmethod
    def _process_index_co(df, index_name):
        df = df.astype(dtype={index_name: int})
        df = df.drop_duplicates(subset=['symbol', index_name])
        return df

    def _prepare_daily_quarterly(self, fields):
        """
        Query and process data from data_api.

        Parameters
        ----------
        fields : list

        Returns
        -------
        merge_d : pd.DataFrame or None
        merge_q : pd.DataFrame or None

        """
        if not fields:
            return None, None

        # query data
        print("Query data - query...")
        daily_list, quarterly_list = self._query_data(self.symbol, fields)

        def pivot_and_sort(df, index_name):
            df = self._process_index_co(df, index_name)
            df = df.pivot(index=index_name, columns='symbol')
            df.columns = df.columns.swaplevel()
            col_names = ['symbol', 'field']
            df.columns.names = col_names
            df = df.sort_index(axis=1, level=col_names)
            df.index.name = index_name
            return df

        multi_daily = None
        multi_quarterly = None
        if daily_list:
            daily_list_pivot = [pivot_and_sort(df, self.TRADE_DATE_FIELD_NAME) for df in daily_list]
            multi_daily = self._merge_data(daily_list_pivot, self.TRADE_DATE_FIELD_NAME)
            # use self.dates as index because original data have weekends
            multi_daily = self._fill_missing_idx_col(multi_daily, index=self.dates, symbols=self.symbol)
            print("Query data - daily fields prepared.")
        if quarterly_list:
            quarterly_list_pivot = [pivot_and_sort(df, self.REPORT_DATE_FIELD_NAME) for df in quarterly_list]
            multi_quarterly = self._merge_data(quarterly_list_pivot, self.REPORT_DATE_FIELD_NAME)
            multi_quarterly = self._fill_missing_idx_col(multi_quarterly, index=None, symbols=self.symbol)
            print("Query data - quarterly fields prepared.")

        # FIXME: patch for lgt_data and rating_data
        fields_lgt_ind = self._get_fields('lgt_data', fields, append=True)
        if fields_lgt_ind:
            multi_daily = self.query_lgt_data(fields_lgt_ind, multi_daily)

        fields_rating_ind = self._get_fields('rating_data', fields, append=True)
        if fields_rating_ind:
            multi_daily = self.query_rating_data(fields_rating_ind, multi_daily)

        tmp_fields = self._get_fields('consensus_data', fields, append=True)
        if tmp_fields:
            multi_daily = self.query_consensus_data(tmp_fields, multi_daily)

        tmp_fields = self._get_fields('stk_rating_data', fields, append=True)
        if tmp_fields:
            multi_daily = self.query_stk_rating_data(tmp_fields, multi_daily)

        tmp_fields = self._get_fields('misc_data', fields, append=True)
        if tmp_fields:
            multi_daily = self.query_misc_data(tmp_fields, multi_daily)

        return multi_daily, multi_quarterly

    def query_rating_data(self, fields_rating_ind, daily_df):
        filter_str = "symbol={0}&start_date={1}&end_date={2}".format(
            ','.join(self.symbol),
            self.extended_start_date_q,
            self.end_date)

        df, msg = self.data_api.query(view='wd.secStockRatingConsus', filter=filter_str)

        df_clean = df[['symbol', 'date', 'wrating_cycle', 'wrating_instnum']]
        df_clean['wrating_cycle'] = df_clean['wrating_cycle'].str[-4:].astype(int) / 1000 * 30
        df_clean['date'] = df_clean['date'].astype(int)
        df_clean = df_clean[df_clean['wrating_cycle'] == 90]
        df_rating = df_clean.pivot_table(values='wrating_instnum', index='date', columns='symbol').fillna(method='ffill')
        df_rating = df_rating.replace(np.nan, 0)

        data_d_orig = self.data_d
        self.data_d = daily_df

        self.append_df(df_rating, 'num_rating', is_quarterly=False)

        daily_df = self.data_d
        self.data_d = data_d_orig
        return daily_df

    def query_stk_rating_data(self, fields_rating_ind, daily_df):
        filter_str = "symbol={0}&start_date={1}&end_date={2}".format(
            ','.join(self.symbol),
            self.extended_start_date_d,
            self.end_date)

        df, msg = self.data_api.query(view='wd.secStockRatingConsus', filter=filter_str)
        if df is None:
            raise ValueError("query wd.secStockRatingConsus Error:" + msg)

        df['date'] = df['date'].astype(int)

        cycle_map = {'263001000' : '30d', '263002000': '90d', '263003000': '180d' }

        data_d_orig = self.data_d
        self.data_d = daily_df

        for cycle, postfix in cycle_map.items():
            df2 = df[df['wrating_cycle']==cycle].copy()
            if df2.empty: continue
            columns = [ (col ,col + '_' + postfix) for col in df2.columns if col not in ['date','symbol', 'wrating_cycle']]
            df2 = df2.rename(columns=dict(columns))
            for col in df2.columns :
                if col in fields_rating_ind and col not in ['date','symbol', 'wrating_cycle']:
                    tmp = df2.pivot_table(values=col, index='date', columns='symbol').fillna(method='ffill')
                    self.append_df(tmp, col, is_quarterly=False)

        daily_df = self.data_d
        self.data_d = data_d_orig
        return daily_df

    def query_consensus_data(self, fields_ind, daily_df):

        rolling_types = set() #['fy0','fy1','fy2','fy3','fttm','yoy','yoy2','cagr']
        query_fields = set(['rolling_type'])
        for f in fields_ind:
            if f in self.consensus_data:
                query_fields.add ( '_'.join(f.split('_')[0:-1]))
                rolling_types.add( f.split('_')[-1] )


        nsymbols  = 20
        data = []
        for i in range((len(self.symbol) + nsymbols -1) // nsymbols):
            symbols = self.symbol[i*nsymbols: (i+1)*nsymbols]

            filter_str = "symbol={0}&start_date={1}&end_date={2}&rolling_type={3}".format(
                ','.join(symbols),
                self.extended_start_date_d,
                self.end_date,
                ','.join(rolling_types).upper()
            )
            # filter_str = "symbol={0}&start_date={1}&end_date={2}".format(
            #     ','.join(symbols),
            #     self.extended_start_date_d,
            #     self.end_date
            # )

            df, msg = self.data_api.query(view='wd.stkConsensusRollingData',
                                          filter=filter_str,
                                          fields=','.join(query_fields))
            if df is None:
                raise ValueError("query wd.stkConsensusRollingData Error:" + msg)
            data.append(df)

        df = pd.concat(data)
        df['date'] = df['date'].astype(int)

        data_d_orig = self.data_d
        self.data_d = daily_df

        for rolling_type in df['rolling_type'].unique():
            df2 = df[df['rolling_type']==rolling_type].copy()
            if df2.empty: continue
            columns = [ (col ,col + '_' + rolling_type.lower()) for col in df2.columns if col not in ['date','symbol', 'rolling_type']]
            df2 = df2.rename(columns=dict(columns))
            df2.index = df2['date']
            for col in df2.columns:
                if col in fields_ind and col not in ['date','symbol', 'wrating_cycle']:
                    tmp = df2.pivot_table(values=col, index='date', columns='symbol').fillna(method='ffill')
                    self.append_df(tmp, col, is_quarterly=False)

        daily_df = self.data_d
        self.data_d = data_d_orig
        return daily_df

    def query_lgt_data(self, fields_lgt_ind, daily_df):
        filter_str = "symbol={0}&start_date={1}&end_date={2}".format(
            ','.join(self.symbol),
            self.extended_start_date_q,
            self.end_date)

        df, msg = self.data_api.query(view="jz.secMoneyFlowNorth",filter=filter_str, fields=','.join(fields_lgt_ind))
        if df is None:
            raise ValueError('query secMoneyFlowNorth error: ' + msg)

        df['trade_date'] = df['trade_date'].astype(np.int32)
        # 持仓数据
        holding_dv = df.pivot_table(index='trade_date', columns='symbol', values='hold_stocks')

        # 持仓比例数据
        holding_ratio = df.pivot_table(index='trade_date', columns='symbol', values='calculate_ratio')

        # 获取除权除息日信息
        df_dividend, msg = self.data_api.query_dividend(','.join(self.symbol), self.extended_start_date_d, self.end_date)
        if df_dividend is None:
            raise ValueError("query_dividend error: " + msg)

        df_dividend = df_dividend[['symbol', 'record_date', 'exdiv_date']]
        df_dividend['values'] = 1
        df_dividend = df_dividend.sort_values(['symbol', 'record_date', 'exdiv_date'])
        df_dividend = df_dividend.dropna()
        df_regdt = df_dividend.pivot_table(index='record_date', columns='symbol', values='values').replace(np.nan, 0)
        df_regdt = df_regdt.replace(np.nan, 0)

        data_d_orig = self.data_d
        self.data_d = daily_df

        # 将除权除息日信息加入DataView
        self.append_df(df_regdt, '_regdt', is_quarterly=False)
        regdt = self.get_ts('_regdt').replace(np.nan, 0.0)
        not_regdt = ~(regdt.astype(bool))

        # 根据股票除权登记日和除权除息日调整lgt_holding
        self.append_df(holding_dv, 'lgt_holding_origin', is_quarterly=False)
        lgt_holding = self.get_ts('lgt_holding_origin')
        adj_lgt_holding = lgt_holding[not_regdt]
        lgt_holding = adj_lgt_holding.fillna(method='ffill')
        self.append_df(lgt_holding, 'lgt_holding', is_quarterly=False)

        # # 根据股票除权登记日和除权除息日调整lgt_holding_ratio
        self.append_df(holding_ratio, 'lgt_holding_ratio_origin', is_quarterly=False)
        lgt_holding_ratio = self.get_ts('lgt_holding_ratio_origin')
        adj_lgt_holding_ratio = lgt_holding_ratio[not_regdt]
        lgt_holding_ratio = adj_lgt_holding_ratio.fillna(method='ffill')
        self.append_df(lgt_holding_ratio, 'lgt_holding_ratio', is_quarterly=False)

        self.remove_field('_regdt,lgt_holding_origin,lgt_holding_ratio_origin')
        daily_df = self.data_d
        self.data_d = data_d_orig
        return daily_df

    def query_misc_data(self, fields, daily_df):

        data_d_orig = self.data_d
        self.data_d = daily_df

        if 'st_flag' in fields:
            df, msg = self.data_api.query(view='wd.stockST')
            if df is None:
                raise ValueError("query wd.secStockRatingConsus Error:" + msg)

            st_symbols = df['symbol'].unique()
            st_daily = self.get_ts('high')
            for symbol in st_daily.columns:
                st_daily[symbol] = ''
                if symbol not in st_symbols: continue
                tmp = df[df['symbol']==symbol]
                for i in range(len(tmp)):
                    x = tmp.iloc[i]
                    entry_date = x['entry_dt']
                    remove_dt  = x['remove_dt']
                    if remove_dt == 0 :
                        remove_dt = 99999999
                    else:
                        remove_dt -= 1
                    stype = x['stype']
                    st_daily.loc[pd.IndexSlice[entry_date: remove_dt], [symbol]] = stype

            self.append_df(st_daily, 'st_flag', is_quarterly=False)

        daily_df = self.data_d
        self.data_d = data_d_orig
        return daily_df

    def _query_data(self, symbol, fields):
        """
        Query data using different APIs, then store them in dict.
        period, start_date and end_date are fixed.
        Keys of dict are securitites.

        Parameters
        ----------
        symbol : list of str
        fields : list of str

        Returns
        -------
        daily_list : list
        quarterly_list : list

        """
        sep = ','
        symbol_str = sep.join(symbol)

        if self.freq == 1:
            daily_list = []
            quarterly_list = []

            # TODO : use fields = {field: kwargs} to enable params
            fields_market_daily = self._get_fields('market_daily', fields, append=True)
            if fields_market_daily:
                print("NOTE: price adjust method is [{:s} adjust]".format(self.adjust_mode))
                # no adjust prices and other market daily fields
                df_daily, msg1 = self.distributed_query('daily', symbol_str,
                                                        start_date=self.extended_start_date_d, end_date=self.end_date,
                                                        adjust_mode=None, fields=sep.join(fields_market_daily),
                                                        limit=100000)
                # df_daily, msg1 = self.data_api.daily(symbol_str, start_date=self.extended_start_date_d, end_date=self.end_date,
                #                                     adjust_mode=None, fields=sep.join(fields_market_daily))

                if self.all_price:
                    adj_cols = ['open', 'high', 'low', 'close', 'vwap']
                    # adjusted prices
                    # df_daily_adjust, msg11 = self.data_api.daily(symbol_str, start_date=self.extended_start_date_d, end_date=self.end_date,
                    #                                             adjust_mode=self.adjust_mode, fields=','.join(adj_cols))
                    df_daily_adjust, msg1 = self.distributed_query('daily', symbol_str,
                                                                   start_date=self.extended_start_date_d,
                                                                   end_date=self.end_date,
                                                                   adjust_mode=self.adjust_mode,
                                                                   fields=sep.join(fields_market_daily), limit=100000)

                    df_daily = pd.merge(df_daily, df_daily_adjust, how='outer',
                                        on=['symbol', 'trade_date'], suffixes=('', '_adj'))
                daily_list.append(df_daily.loc[:, fields_market_daily])

            fields_ref_daily = self._get_fields('ref_daily', fields, append=True)
            if fields_ref_daily:
                df_ref_daily, msg2 = self.distributed_query('query_lb_dailyindicator', symbol_str,
                                                            start_date=self.extended_start_date_d,
                                                            end_date=self.end_date,
                                                            fields=sep.join(fields_ref_daily), limit=20000)
                daily_list.append(df_ref_daily.loc[:, fields_ref_daily])

            fields_income = self._get_fields('income', fields, append=True)
            if fields_income:
                df_income, msg3 = self.data_api.query_lb_fin_stat('income', symbol_str, self.extended_start_date_q,
                                                                  self.end_date,
                                                                  sep.join(fields_income),
                                                                  drop_dup_cols=['symbol', self.REPORT_DATE_FIELD_NAME])
                quarterly_list.append(df_income.loc[:, fields_income])

            fields_balance = self._get_fields('balance_sheet', fields, append=True)
            if fields_balance:
                df_balance, msg3 = self.data_api.query_lb_fin_stat('balance_sheet', symbol_str,
                                                                   self.extended_start_date_q, self.end_date,
                                                                   sep.join(fields_balance), drop_dup_cols=['symbol',
                                                                                                            self.REPORT_DATE_FIELD_NAME])
                quarterly_list.append(df_balance.loc[:, fields_balance])

            fields_cf = self._get_fields('cash_flow', fields, append=True)
            if fields_cf:
                df_cf, msg3 = self.data_api.query_lb_fin_stat('cash_flow', symbol_str, self.extended_start_date_q,
                                                              self.end_date,
                                                              sep.join(fields_cf),
                                                              drop_dup_cols=['symbol', self.REPORT_DATE_FIELD_NAME])
                quarterly_list.append(df_cf.loc[:, fields_cf])

            fields_fin_ind = self._get_fields('fin_indicator', fields, append=True)
            if fields_fin_ind:
                df_fin_ind, msg4 = self.data_api.query_lb_fin_stat('fin_indicator', symbol_str,
                                                                   self.extended_start_date_q, self.end_date,
                                                                   sep.join(fields_fin_ind), drop_dup_cols=['symbol',
                                                                                                            self.REPORT_DATE_FIELD_NAME])
                quarterly_list.append(df_fin_ind.loc[:, fields_fin_ind])

            fields_risk_model = self._get_fields('risk_model', fields, append=True)    
            if fields_risk_model:
                df_risk_model, msg5 = self.data_api.query_risk_model( symbol_str,
                                                                   self.extended_start_date_q, self.end_date,
                                                                   sep.join(fields_risk_model))
                daily_list.append(df_risk_model.loc[:, fields_risk_model])                  

        else:
            raise NotImplementedError("freq = {}".format(self.freq))

        return daily_list, quarterly_list

    @staticmethod
    def _merge_data(dfs, index_name='trade_date', join='outer', keep_input=True):
        """
        Merge data from different APIs into one DataFrame.

        Parameters
        ----------
        dfs : list of pd.DataFrame

        Returns
        -------
        merge : pd.DataFrame or None
            If dfs is empty, return None

        Notes
        -----
        Align on date index, concatenate on columns (symbol and fields)

        """
        # dfs = [df for df in dfs if df is not None]
        new_dfs = []
        # column level swap: [symbol, field] => [field, symbol]
        for df in dfs:
            if keep_input:
                df_new = df.copy()
            else:
                df_new = df
            df_new.columns = df_new.columns.swaplevel()
            df_new = df_new.sort_index(axis=1)
            new_dfs.append(df_new)

        index_set = None
        for df in new_dfs:
            if index_set is None:
                index_set = set(df.index)
            else:
                if join == 'inner':
                    index_set = index_set & set(df.index)
                else:
                    index_set = index_set | set(df.index)
        index_list = list(index_set)
        index_list.sort()
            
        cols = None
        for df in new_dfs:
            if cols is None:
                cols = df.columns
            else:
                cols = cols.append(df.columns)
        
        merge = pd.DataFrame(data=np.nan, index=index_list, columns=cols)
        
        for df in new_dfs:
            for col in df.columns.levels[0]:
                if not df[col].empty:
                    merge[col] = df[col]
        
        merge.columns = merge.columns.swaplevel()
        merge = merge.sort_index(axis=1)
    
        # merge1 = pd.concat(dfs, axis=1, join='outer')
        # drop duplicated columns. ONE LINE EFFICIENT version
        mask_duplicated = merge.columns.duplicated()
        if np.any(mask_duplicated):
            # print("Duplicated columns found. Dropped.")
            merge = merge.loc[:, ~mask_duplicated]

            # if merge.isnull().sum().sum() > 0:
            # print "WARNING: nan in final merged data. NO fill"
            # merge.fillna(method='ffill', inplace=True)

        merge = merge.sort_index(axis=1, level=['symbol', 'field'])
        merge.index.name = index_name

        return merge

    def _fill_missing_idx_col(self, df, index=None, symbols=None):
        if index is None:
            index = df.index
        if symbols is None:
            symbols = self.symbol
        fields = df.columns.levels[1]

        if len(fields) * len(self.symbol) != len(df.columns) or set(index) != set(df.index):
            cols_multi = pd.MultiIndex.from_product([fields, symbols], names=['field', 'symbol'])
            cols_multi = cols_multi.sort_values()

            df_final = pd.DataFrame(index=index, columns=cols_multi, data=np.nan)
            df_final.index.name = df.index.name
            
            df.columns = df.columns.swaplevel()
            df = df.sort_index(axis=1)
            
            for col in df.columns.levels[0]:
                df_final[col] = df[col]
            
            df_final.columns = df_final.columns.swaplevel()
            df_final = df_final.sort_index(axis=1)
            # df_final.update(df)
        
            # idx_diff = sorted(set(df_final.index) - set(df.index))
            col_diff = sorted(set(df_final.columns.levels[0].values) - set(df.columns.levels[0].values))
            print("WARNING: some data is unavailable: "
                   # + "\n    At index " + ', '.join(idx_diff)
                   + "\n    At fields " + ', '.join(col_diff))
            return df_final
        else:
            return df

    def _align_and_merge_q_into_d(self):
        data_d, data_q = self.data_d, self.data_q
        if data_d is not None and data_q is not None:
            df_ref_ann = data_q.loc[:, pd.IndexSlice[:, self.ANN_DATE_FIELD_NAME]].copy()
            df_ref_ann.columns = df_ref_ann.columns.droplevel(level='field')

            dic_expanded = dict()
            for field_name, df in data_q.groupby(level=1, axis=1):  # by column multiindex fields
                df_expanded = align(df, df_ref_ann, self.dates)
                dic_expanded[field_name] = df_expanded
            df_quarterly_expanded = pd.concat(dic_expanded.values(), axis=1)
            df_quarterly_expanded.index.name = self.TRADE_DATE_FIELD_NAME

            data_d_merge = self._merge_data([data_d, df_quarterly_expanded], index_name=self.TRADE_DATE_FIELD_NAME)
            data_d = data_d_merge.loc[data_d.index, :]
        self.data_d = data_d

    def _prepare_adj_factor(self):
        """Query and append daily adjust factor for prices."""
        mask_stocks = self.data_inst['inst_type'] == 1
        if mask_stocks.sum() == 0:
            return
        symbol_stocks = self.data_inst.loc[mask_stocks].index.values
        symbol_str = ','.join(symbol_stocks)
        df_adj = self.data_api.query_adj_factor_daily(symbol_str,
                                                      start_date=self.extended_start_date_d, end_date=self.end_date,
                                                      div=False)
        self.append_df(df_adj, 'adjust_factor', is_quarterly=False)

    def _prepare_comp_info(self):
        # if a symbol is index member of any one universe, its value of index_member will be 1.0

        #  Backup code
        #    Multiple universe
        #     res = dict()
        #     for univ in self.universe:
        #         df = self.data_api.query_index_member_daily(univ, self.extended_start_date_d, self.end_date)
        #         res[univ] = df
        #     df_res = pd.concat(res, axis=0)
        #     df = df_res.groupby(by='trade_date').apply(lambda df: df.any(axis=0)).astype(float)
        #
        #     # Always include additional symbols
        #     # for code in self.symbol:
        #     #     if code not in df.columns:
        #     #         df[code] = 1.0
        # else:
        # tzxu 2018.5.28 only set index_member and index_weight of first universe
        first_universe = self.universe[0]
        if first_universe.endswith(".UNI"):
            df = self.data_api.query_universe_member_daily(first_universe, self.extended_start_date_d, self.end_date)
            df = df.groupby(by='trade_date').apply(lambda df: df.any(axis=0)).astype(float)
            self.append_df(df, 'index_member', is_quarterly=False)

            # use weights of the first universe
            df_weights = self.data_api.query_universe_weights_daily(first_universe, self.extended_start_date_d,
                                                                    self.end_date)
            self.append_df(df_weights, 'index_weight', is_quarterly=False)
        else:
            df = self.data_api.query_index_member_daily(first_universe, self.extended_start_date_d, self.end_date)
            df = df.groupby(by='trade_date').apply(lambda df: df.any(axis=0)).astype(float)
            self.append_df(df, 'index_member', is_quarterly=False)

            # use weights of the first universe
            df_weights = self.data_api.query_index_weights_daily(first_universe, self.extended_start_date_d,
                                                                 self.end_date)

            if first_universe not in self.index_weights:
                self.index_weights[first_universe] = df_weights

            self.append_df(df_weights, 'index_weight', is_quarterly=False)

    def _prepare_report_date(self):
        idx = self.data_q.index
        df_report_date = pd.DataFrame(index=idx, columns=self.symbol, data=0)
        n = len(idx)
        quarter = idx.values // 100 % 100
        df_report_date.loc[:, :] = quarter.reshape(n, -1)

        self.append_df(df_report_date, 'quarter', is_quarterly=True)

    def _prepare_inst_info(self):
        res = self.data_api.query_inst_info(symbol=','.join(self.symbol),
                                            fields='symbol,inst_type,name,list_date,'
                                                   'delist_date,product,pricetick,multiplier,'
                                                   'buylot,setlot',
                                            inst_type="")
        self._data_inst = res

    def _prepare_group(self, group_fields):
        data_map = {'sw1': ('SW', 1),
                    'sw2': ('SW', 2),
                    'sw3': ('SW', 3),
                    'sw4': ('SW', 4),
                    'zz1': ('ZZ', 1),
                    'zz2': ('ZZ', 2)}
        for field in group_fields:
            type_, level = data_map[field]
            df = self.data_api.query_industry_daily(symbol=','.join(self.symbol),
                                                    start_date=self.extended_start_date_q, end_date=self.end_date,
                                                    type_=type_, level=level)
            self.append_df(df, field, is_quarterly=False)
            self.industry_groups[field] = df

    def _prepare_benchmark(self):
        if self.benchmark == 'VW_UNIVERSE':
            df_close = self.get_ts('close_adj', start_date=self.extended_start_date_d)
            df_ret   = df_close.pct_change()
            df_float_mv = self.get_ts('float_mv', start_date=self.extended_start_date_d)
            df_weight = df_float_mv.div(df_float_mv.sum(axis = 1), axis = 0)
            df_ret_vw = df_weight.mul(df_ret).sum(axis = 1).replace(np.nan, 0.0)
            df_price = (1 + df_ret_vw).cumprod()
            df_price = pd.DataFrame(df_price)
            df_price.columns = ['close']
            df_bench = df_price
        elif self.benchmark == 'EW_UNIVERSE':
            df_close = self.get_ts('close_adj', start_date=self.extended_start_date_d)
            df_ret   = df_close.pct_change()
            df_ret_ew = df_ret.mean(axis = 1).replace(np.nan, 0.0)
            df_price = (1 + df_ret_ew).cumprod()
            df_price = pd.DataFrame(df_price)
            df_price.columns = ['close']
            df_bench = df_price
        else:
            df_bench, msg = self.data_api.daily(self.benchmark,
                                                start_date=self.extended_start_date_d, end_date=self.end_date,
                                                adjust_mode=self.adjust_mode,
                                                fields='trade_date,symbol,open,high,low,close,vwap,volume,turnover')
            # TODO: we want more than just close price of benchmark
            df_bench = df_bench.set_index('trade_date').loc[:, ['open','high','low','close']]

            is_index = re.match('399.*.SZ', self.benchmark) or re.match('000.*.SH', self.benchmark)
            if is_index:
                # use weights of the first universe
                if self.benchmark not in self.index_weights:
                    df_weights = self.data_api.query_index_weights_daily(self.benchmark, self.extended_start_date_d,
                                                                     self.end_date)
                    self.index_weights[self.benchmark] = df_weights

            # Add bm_high, bm_low, bm_open, bm_close to each code
            tmp = self.get_ts('open')
            for field in ['open','high','low','close']:
                for symbol in tmp.columns:
                    tmp[symbol] = df_bench[field]
                self.append_df(tmp,   'bm_' + field,  is_quarterly=False)

            df_bench = df_bench[['close']]
        return df_bench

    # --------------------------------------------------------------------------------------------------------
    # Add/Remove Fields&Formulas
    def _add_field(self, field_name, is_quarterly=None):
        if field_name in self.fields:
            return

        self.fields.append(field_name)
        if not self._is_predefined_field(field_name):
            if is_quarterly is None:
                raise ValueError("Field [{:s}] is not a predefined field, but no frequency information is provided.")
            if is_quarterly:
                self.custom_quarterly_fields.append(field_name)
            else:
                self.custom_daily_fields.append(field_name)

    def add_field(self, field_name, data_api=None):
        """
        Query and append new field to DataView.

        Parameters
        ----------
        data_api : BaseDataServer
        field_name : str
            Must be a known field name (which is given in documents).

        Returns
        -------
        bool
            whether add successfully.

        """
        if data_api is None:
            if self.data_api is None:
                print("Add field failed. No data_api available. Please specify one in parameter.")
                return False
        else:
            self.data_api = data_api

        if field_name in self.fields:
            print("Field name [{:s}] already exists.".format(field_name))
            return False

        if not self._is_predefined_field(field_name):
            print("Field name [{}] not valid, ignore.".format(field_name))
            return False

        merge_d, merge_q = self._prepare_daily_quarterly([field_name])

        if self._is_daily_field(field_name):
            if self.data_d is None:
                raise ValueError("Please prepare [{:s}] first.".format(field_name))
            merge, _ = self._prepare_daily_quarterly([field_name])
            is_quarterly = False
        else:
            if self.data_q is None:
                raise ValueError("Please prepare [{:s}] first.".format(field_name))
            _, merge = self._prepare_daily_quarterly([field_name])
            is_quarterly = True

        merge = merge.loc[:, pd.IndexSlice[:, field_name]]
        merge.columns = merge.columns.droplevel(level=1)
        self.append_df(merge, field_name, is_quarterly=is_quarterly)  # whether contain only trade days is decided by existing data.
        
        if is_quarterly:
            df_ann = merge_q.loc[:, pd.IndexSlice[:, self.ANN_DATE_FIELD_NAME]]
            df_ann.columns = df_ann.columns.droplevel(level='field')
            df_expanded = align(merge, df_ann, self.dates)
            self.append_df(df_expanded, field_name, is_quarterly=False)
        return True

    def _create_parser(self, formula_func_name_style='camel', allow_future_data=False):
        parser = Parser(allow_future_data=allow_future_data)
        parser.set_capital(formula_func_name_style)

        for key in self._import_factors:
            factor = self._import_factors[key]
            parser.register_function(factor.name, FactorFunc(self, factor, allow_future_data))

        parser.register_function("ResReturn", ResReturnFunc(self, allow_future_data))
        return parser

    def _get_var(self, var, allow_future_data):

        if not allow_future_data:
            if var in self.labels:
                raise ValueError("Variable {0} is label while calculating factor!".format(var))

        if var in self._import_factors:
            if self._is_quarter_field(var):
                df_var = self.get_ts_quarter(var, start_date=self.extended_start_date_q)
            elif var in self.fields:
                df_var = self.get_ts(var, start_date=self.extended_start_date_d, end_date=self.end_date)
            else:
                factor_def = self._import_factors[var]
                df_var = FactorFunc(self, factor_def, allow_future_data)()
                self.append_df(df_var, var, is_quarterly=factor_def.is_quarterly)

            return df_var

        if self._is_quarter_field(var):
            return self.get_ts_quarter(var, start_date=self.extended_start_date_q)
        else:
            # must use extended date. Default is start_date
            return self.get_ts(var, start_date=self.extended_start_date_d, end_date=self.end_date)

    def add_factor(self, factor, name=None, is_quarterly=False):  # within_index=True):
        if not name:
            name = factor.split('(')[0]

        self.add_formula(field_name=name, formula=factor, is_quarterly=is_quarterly, is_factor=True)

    def add_label(self, factor, name=None, is_quarterly=False):  # within_index=True):
        if not name:
            name = factor.split('(')[0]

        self.add_formula(field_name=name, formula=factor, is_quarterly=is_quarterly, is_factor=False)

    def add_formula(self, field_name, formula, is_quarterly, overwrite=True,
                    formula_func_name_style='camel', data_api=None,
                    within_index=True, is_factor=True):
        """
        Add a new field, which is calculated using existing fields.

        Parameters
        ----------
        formula : str or unicode
            A formula contains operations and function calls.
        field_name : str or unicode
            A custom name for the new field.
        is_quarterly : bool
            Whether df is quarterly data (like quarterly financial statement) or daily data.
        overwrite : bool, optional
            Whether overwrite existing field. True by default.
        formula_func_name_style : {'upper', 'lower'}, optional
        data_api : RemoteDataService, optional
        within_index : bool
            When do cross-section operatioins, whether just do within index components.
        is_factor: bool
            Whether new field is factor or label.

        Notes
        -----
        Time cost of this function:
            For a simple formula (like 'a + 1'), almost all time is consumed by append_df;
            For a complex formula (like 'GroupRank'), half of time is consumed by evaluation and half by append_df.
        """
        if data_api is not None:
            self.data_api = data_api

        if field_name in self.fields:
            if overwrite:
                self.remove_field(field_name)
                print("Field [{:s}] is overwritten.".format(field_name))
            else:
                raise ValueError("Add formula failed: name [{:s}] exist. Try another name.".format(field_name))
        elif self._is_predefined_field(field_name):
            raise ValueError("[{:s}] is alread a pre-defined field. Please use another name.".format(field_name))

        parser = self._create_parser(formula_func_name_style, allow_future_data=not is_factor)
        expr = parser.parse(formula)

        var_df_dic = dict()
        var_list = expr.variables()

        factors = [var for var in var_list if var in self._import_factors]

        var_list = [var for var in var_list if var not in expr.functions]

        # TODO: users do not need to prepare data before add_formula
        if not self.fields:
            self.fields.extend(var_list)
            self.prepare_data()
        else:
            for var in var_list:
                if var not in self.fields:
                    print("Variable [{:s}] is not recognized (it may be wrong)," \
                          "try to fetch from the server...".format(var))
                    success = self.add_field(var)
                    if not success:
                        return
        if is_factor:
            for var in var_list:
                if var in self.labels:
                    raise ValueError("Variable {0} is label while calculating factor!".format(var))
        else:
            if field_name not in self.labels:
                self.labels.append(field_name)

        for var in var_list:
            if self._is_quarter_field(var):
                df_var = self.get_ts_quarter(var, start_date=self.extended_start_date_q)
            else:
                # must use extended date. Default is start_date
                df_var = self.get_ts(var, start_date=self.extended_start_date_d, end_date=self.end_date)

            var_df_dic[var] = df_var

        # TODO:
        for factor in factors:
            if factor in self.fields:
                df_var = self.get_ts(factor, start_date=self.extended_start_date_d, end_date=self.end_date)
                var_df_dic[factor] = df_var

            elif not self._import_factors[factor].args:
                f = parser.functions[factor]
                df_var = f()
                self.append_df(df_var, factor, is_quarterly=is_quarterly)
                var_df_dic[factor] = df_var
            # else:
            #     raise ValueError("no arguments for factor: " + factor)

        # TODO: send ann_date into expr.evaluate. We assume that ann_date of all fields of a symbol is the same
        df_ann = self._get_ann_df()
        if within_index:
            df_index_member = self.get_ts('index_member', start_date=self.extended_start_date_d, end_date=self.end_date)
            df_eval = parser.evaluate(var_df_dic, ann_dts=df_ann, trade_dts=self.dates, index_member=df_index_member)
        else:
            df_eval = parser.evaluate(var_df_dic, ann_dts=df_ann, trade_dts=self.dates)

        # FIXME: When field_name is factor's id, the factor may be added before!
        if field_name in self.fields:
            self.remove_field(field_name)

        self.append_df(df_eval, field_name, is_quarterly=is_quarterly)

        if is_quarterly:
            df_ann = self._get_ann_df()
            df_expanded = align(df_eval, df_ann, self.dates)
            self.append_df(df_expanded, field_name, is_quarterly=False)

    def append_df(self, df, field_name, is_quarterly=False):
        """
        Append DataFrame to existing multi-index DataFrame and add corresponding field name.

        Parameters
        ----------
        df : pd.DataFrame or pd.Series
        field_name : str or unicode
        is_quarterly : bool
            Whether df is quarterly data (like quarterly financial statement) or daily data.

        Notes
        -----
        append_df does not support overwrite. To overwrite a field, you must first do self.remove_fields(),
        then append_df() again.

        """
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            pass
        elif isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        else:
            raise ValueError("Data to be appended must be pandas format. But we have {}".format(type(df)))

        if is_quarterly:
            the_data = self.data_q
        else:
            the_data = self.data_d

        # Copy exists symbols and set multi index
        df = df.loc[:, the_data.columns.levels[0]]
        multi_idx = pd.MultiIndex.from_product([[field_name], df.columns])
        df.columns = multi_idx
        df = df.sort_index(axis=1)

        the_data.columns = the_data.columns.swaplevel()
        the_data = the_data.sort_index(axis=1)
        
        new_cols = the_data.columns.append(df.columns)
        the_data = the_data.reindex(columns=new_cols)
        the_data[field_name] = df[field_name]
        the_data.columns = the_data.columns.swaplevel()
        the_data = the_data.sort_index(axis=1)
        #the_data = apply_in_subprocess(pd.merge, args=(the_data, df),
        #                            kwargs={'left_index': True, 'right_index': True, 'how': 'left'})  # runs in *only* one process
        #the_data = pd.merge(the_data, df, left_index=True, right_index=True, how='left')
        #the_data = the_data.sort_index(axis=1)
        #merge = the_data.join(df, how='left')  # left: keep index of existing data unchanged
        #sort_columns(the_data)
    
        if is_quarterly:
            self.data_q = the_data
        else:
            self.data_d = the_data
        self._add_field(field_name, is_quarterly)

    def remove_field(self, field_names):
        """
        Query and append new field to DataView.

        Parameters
        ----------
        field_names : str
            Separated by comma.
            The (custom) field to be removed from dataview.

        Returns
        -------
        bool
            whether add successfully.

        """
        if isinstance(field_names, basestring):
            field_names = field_names.split(',')
        else:
            raise ValueError("field_names must be str separated by comma.")

        for field_name in field_names:
            # parameter validation
            if field_name not in self.fields:
                print("Field name [{:s}] does not exist. Stop remove_field.".format(field_name))
                return

            if self._is_daily_field(field_name):
                is_quarterly = False
            elif self._is_quarter_field(field_name):
                is_quarterly = True
            else:
                print("Field name [{}] is a pre-defined field, ignore.".format(field_name))
                return

            # remove field data

            self.data_d = self.data_d.drop(field_name, axis=1, level=1)
            if is_quarterly:
                self.data_q = self.data_q.drop(field_name, axis=1, level=1)

            # remove fields name from list
            self.fields.remove(field_name)
            if is_quarterly:
                if field_name in self.custom_quarterly_fields:
                    self.custom_quarterly_fields.remove(field_name)
            else:
                if field_name in self.custom_daily_fields:
                    self.custom_daily_fields.remove(field_name)

    # --------------------------------------------------------------------------------------------------------
    # Get Data API
    def get(self, symbol="", start_date=0, end_date=0, fields=""):
        """
        Basic API to get arbitrary data. If nothing fetched, return None.

        Parameters
        ----------
        symbol : str, optional
            Separated by ',' default "" (all securities).
        start_date : int, optional
            Default 0 (self.start_date).
        end_date : int, optional
            Default 0 (self.start_date).
        fields : str, optional
            Separated by ',' default "" (all fields).

        Returns
        -------
        res : pd.DataFrame or None
            index is datetimeindex, columns are (symbol, fields) MultiIndex

        """
        sep = ','

        if not fields:
            fields = slice(None)  # self.fields
        else:
            fields = fields.split(sep)

        if not symbol:
            symbol = slice(None)  # this is 3X faster than symbol = self.symbol
        else:
            symbol = symbol.split(sep)

        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date

        res = self.data_d.loc[pd.IndexSlice[start_date: end_date], pd.IndexSlice[symbol, fields]]
        return res

    def get_snapshot(self, snapshot_date, symbol="", fields=""):
        """
        Get snapshot of given fields and symbol at snapshot_date.

        Parameters
        ----------
        snapshot_date : int
            Date of snapshot.
        symbol : str, optional
            Separated by ',' default "" (all securities).
        fields : str, optional
            Separated by ',' default "" (all fields).

        Returns
        -------
        res : pd.DataFrame
            symbol as index, field as columns

        """

        if self._snapshot is not None:
            if snapshot_date not in self._snapshot:
                return

            df = self._snapshot[snapshot_date]
            if fields:
                return df[fields.split(',')]
            else:
                return df

        res = self.get(symbol=symbol, start_date=snapshot_date, end_date=snapshot_date, fields=fields)
        if res is None:
            print("No data. for date={}, fields={}, symbol={}".format(snapshot_date, fields, symbol))
            return

        res = res.stack(level='symbol', dropna=False)
        res.index = res.index.droplevel(level=self.TRADE_DATE_FIELD_NAME)

        return res

    def _get_ann_df(self):
        """
        Query announcement date of financial statements of all securities.

        Returns
        -------
        df_ann : pd.DataFrame or None
            Index is date, column is symbol.
            If no quarterly data available, return None.

        """
        if self.data_q is None:
            return None
        df_ann = self.data_q.loc[:, pd.IndexSlice[:, self.ANN_DATE_FIELD_NAME]]
        df_ann.columns = df_ann.columns.droplevel(level='field')

        return df_ann

    def get_symbol(self, symbol, start_date=0, end_date=0, fields=""):
        res = self.get(symbol, start_date=start_date, end_date=end_date, fields=fields)
        if res is None:
            raise ValueError("No data. for "
                             "start_date={}, end_date={}, field={}, symbol={}".format(start_date, end_date,
                                                                                      fields, symbol))

        res.columns = res.columns.droplevel(level='symbol')
        return res

    def get_ts_quarter(self, fields, symbols="", start_date=0, end_date=0):
        # TODO
        sep = ','
        if not symbols:
            symbols = self.symbol
        else:
            symbols = symbols.split(sep)

        if not fields:
            fields = slice(None)  # self.fields
        else:
            fields = fields.split(sep)

        # if not start_date:
        #     start_date = self.start_date
        # if not end_date:
        #     end_date = self.end_date

        df_ref_quarterly = self.data_q.loc[:, pd.IndexSlice[symbols, fields]]
        df_ref_quarterly.columns = df_ref_quarterly.columns.droplevel(level='field')

        return df_ref_quarterly

    def get_ts(self, field, symbol="", start_date=0, end_date=0, keep_level=False):
        """
        Get time series data of single field.

        Parameters
        ----------
        field : str or unicode
            Single field.
        symbol : str, optional
            Separated by ',' default "" (all securities).
        start_date : int, optional
            Default 0 (self.start_date).
        end_date : int, optional
            Default 0 (self.start_date).

        Returns
        -------
        res : pd.DataFrame
            Index is int date, column is symbol.

        """
        res = self.get(symbol, start_date=start_date, end_date=end_date, fields=field)
        if res is None:
            print("No data. for start_date={}, end_date={}, field={}, symbol={}".format(start_date,
                                                                                        end_date, field, symbol))
            raise ValueError
            return

        #if not keep_level and len(res.columns) and len(field.split(',')) == 1:
        if not keep_level and len(field.split(',')) == 1:
            res.columns = res.columns.droplevel(level='field')
            # XXX Save field name for ResReturnFunc
            res.columns.name = field

        return res

    # --------------------------------------------------------------------------------------------------------
    # DataView I/O
    @staticmethod
    def _load_h5(fp):
        """Load data and meta_data from hd5 file.

        Parameters
        ----------
        fp : str, optional
            File path of pre-stored hd5 file.

        """
        h5 = pd.HDFStore(fp)

        res = dict()
        for key in h5.keys():
            res[key] = h5.get(key)

        h5.close()

        return res

    def _process_data(self, large_memory=False):
        """
        Process data for improving performance, preload factors
        """
        for factor in self.factors + self.labels:
            factor = factor.strip()
            if not factor: continue

            factor_name = ""
            factor_expr = ""
            if "=" in factor:
                tmp = factor.split('=')
                factor_name = tmp[0]
                factor_expr = tmp[1]
            else:
                factor_name = factor
                factor_expr = factor

            factor_id = factor_expr.split('(')[0].strip()
            if factor_id not in self._import_factors:
                #print("Can't find factor definitions: " + factor_id)
                continue
            if self._import_factors[factor_id].is_quarterly:
                t = self.get_ts_quarter(factor_name)
                if t is None or len(t.columns) == 0:
                    self.add_factor(factor_expr, factor_name, is_quarterly=True)
            else:
                t = self.get_ts(factor_name)
                if t is None or len(t.columns) == 0:
                    self.add_factor(factor_expr, factor_name, is_quarterly=False)

        t = self.get_ts('_daily_adjust_factor')
        if t is None or len(t.columns) == 0:
            a = self.get_ts('adjust_factor')
            b = (a / a.shift(1)).fillna(1.0)
            self.append_df(b, '_daily_adjust_factor', is_quarterly=False)

        t = self.get_ts("_limit")
        if t is None or len(t.columns) == 0:
            dates = self.dates
            mask = dates < self.start_date
            before_first_day = dates[mask][-1]

            open = self.get_ts('open_adj')
            preclose = self.get_ts('close_adj', start_date=before_first_day).shift(1)
            limit = np.abs((open - preclose) / preclose)
            self.append_df(limit, "_limit", is_quarterly=False)

        # Snapshot dict may use large memory.

        if large_memory:
            self.update_snapshot()

    def update_snapshot(self):
        dates = self.data_d.index.values
        df = self.data_d.T.unstack()
        self._snapshot = {}
        for date in dates:
            tmp = df[date].copy()
            del tmp.index.name
            del tmp.columns.name
            self._snapshot[date] = tmp

    def load_dataview(self, folder_path='.', large_memory=True):
        """
        Load data from local file.

        Parameters
        ----------
        folder_path : str or unicode, optional
            Folder path to store hd5 file and meta data.

        """

        path_meta_data = os.path.join(folder_path, 'meta_data.json')
        path_data = os.path.join(folder_path, 'data.hd5')
        if not (os.path.exists(path_meta_data) and os.path.exists(path_data)):
            raise IOError("There is no data file under directory {}".format(folder_path))

        meta_data = jutil.read_json(path_meta_data)
        dic = self._load_h5(path_data)
        self.data_d = dic.get('/data_d', None)
        self.data_q = dic.get('/data_q', None)
        self._data_benchmark = dic.get('/data_benchmark', None)
        self._data_inst = dic.get('/data_inst', None)
        self._factor_df = dic.get('/factor_df', None)

        for k in dic.keys():
            if k.startswith('/index_weight/'):
                self.index_weights[k.split('/')[2]] = dic[k]
            if k.startswith('/industry_group/'):
                self.industry_groups[k.split('/')[2]] = dic[k]

        self.__dict__.update(meta_data)

        for index, row in self._factor_df.iterrows():
            factor_id = row['factor_id']
            factor_body = row['factor_def']
            factor_args = list(filter(None, row['factor_args'].split(',')))
            if 'factor_quarterly' in row:
                factor_quarterly = row['factor_quarterly']
            else:
                factor_quarterly = False

            self._import_factors[factor_id] = FactorDef(factor_id, factor_args, factor_body, factor_quarterly)

        self._process_data(large_memory)

        print("Dataview loaded successfully.")

    def save_dataview(self, folder_path):
        """
        Save data and meta_data_to_store to a single hd5 file.
        Store at output/sub_folder

        Parameters
        ----------
        folder_path : str or unicode
            Path to store your data.

        """
        abs_folder = os.path.abspath(folder_path)
        meta_path = os.path.join(folder_path, 'meta_data.json')
        data_path = os.path.join(folder_path, 'data.hd5')

        data_to_store = {'data_d': self.data_d,
                         'data_q': self.data_q,
                         'data_benchmark': self.data_benchmark,
                         'data_inst': self.data_inst,
                         'factor_df': self._factor_df
                         }
        for symbol in self.index_weights.keys():
            data_to_store['index_weight/' + symbol] = self.index_weights[symbol]

        for group in self.industry_groups.keys():
            data_to_store['industry_group/' + group] = self.industry_groups[group]

        data_to_store = {k: v for k, v in data_to_store.items() if v is not None}
        meta_data_to_store = {key: self.__dict__[key] for key in self.meta_data_list}


        print("\nStore data...")
        jutil.save_json(meta_data_to_store, meta_path)
        self._save_h5(data_path, data_to_store)

        print("Dataview has been successfully saved to:\n"
              + abs_folder + "\n\n"
              + "You can load it with load_dataview('{:s}')".format(abs_folder))

    @staticmethod
    def _save_h5(fp, dic):
        """
        Save data in dic to a hd5 file.

        Parameters
        ----------
        fp : str
            File path.
        dic : dict

        """
        import warnings
        warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

        jutil.create_dir(fp)
        h5 = pd.HDFStore(fp, complevel=9, complib='blosc')
        for key, value in dic.items():
            h5[key] = value
        h5.close()

    def dup(self, symbols=None, remove_fields=None, start_date=None, end_date=None, fields=None, with_default_fields=True, large_memory=False):
        """
        Duplicate this dataview with less symbols, dates between start_date and end_date and less fields.

        Parameters
        ----------
        symbols:       only copy data of symbols in this parameter.
        remove_fields: only copy fields not in this parameter.
        start_date:    only copy data starting with this date.
        end_date:      only copy data ending with this date.
        fields:        only copy fields in this parameter.
        """
        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date

        extended_start_date_d = jutil.shift(self.start_date, n_weeks=-8)  # query more data
        extended_start_date_q = jutil.shift(self.start_date, n_weeks=-80)

        if remove_fields and fields:
            raise ValueError("Shouldn't use both fields and remove_fields")

        if remove_fields:
            if isinstance(remove_fields, basestring):
                remove_fields = remove_fields.split(',')
            exist_fields = self.fields
            fields = [x for x in exist_fields if x not in remove_fields]
        elif fields:
            if isinstance(fields, basestring):
                fields = fields.split(',')
        else:
            fields = slice(None)

        if with_default_fields:
            if fields != slice(None):
                fields = list( set(fields) | set(self.default_fields))
        print("dup fields:", fields)

        if not symbols:
            symbols = slice(None)
        elif isinstance(symbols, basestring):
            symbols = symbols.split(',')
        elif isinstance(symbols, (list, tuple)):
            pass

        dv2 = DataView()
        dv2.data_benchmark = self.data_benchmark.loc[pd.IndexSlice[extended_start_date_d: end_date]]
        if self.data_d is not None:
            dv2.data_d = self.data_d.loc[pd.IndexSlice[extended_start_date_d: end_date], pd.IndexSlice[symbols, fields]]
        if self.data_q is not None:
            dv2.data_q = self.data_q.loc[pd.IndexSlice[extended_start_date_d: end_date], pd.IndexSlice[symbols, slice(None)]]

        dv2._data_inst = self.data_inst.copy()

        meta_data = {key: self.__dict__[key] for key in self.meta_data_list}
        meta_data['start_date'] = start_date
        meta_data['end_date'] = end_date
        if symbols != slice(None):
            meta_data['symbol'] = sorted(set(meta_data['symbol']) & set(symbols))

        if fields != slice(None):
            meta_data['fields'] = fields
        dv2.__dict__.update(meta_data)

        dv2.index_weights = {}
        for k,v in self.index_weights.items():
            dv2.index_weights[k] = v[ (v.index >= extended_start_date_d) & (v.index <= end_date)]

        dv2.industry_groups = {}
        for k,v in self.index_weights.items():
            dv2.industry_groups[k] = v[(v.index >= extended_start_date_d) & (v.index <= end_date)]

        dv2._process_data(large_memory=large_memory)
        return dv2

    @staticmethod
    def concat(dv1, dv2):
        def concat_df(dv1, dv2, name):
            a = dv1.__dict__[name]
            b = dv2.__dict__[name]
            if a is None:
                return b
            elif b is None:
                return a
            else:
                a = a[a.index < dv2.start_date]
                b = b[b.index > dv1.end_date]
                return pd.concat([a, b])

        dv = DataView()
        for key in dv.meta_data_list:
            dv.__dict__[key] = dv1.__dict__[key]

        dv.end_date = dv2.end_date
        for key in ['fields', 'symbol', 'custom_daily_fields', 'custom_quarterly_fields',
                    'factors', 'load_factors', 'labels', 'load_labels']:
            dv.__dict__[key] = list(set(dv1.__dict__[key] + dv2.__dict__[key]))

        dv._data_inst = pd.concat([dv1._data_inst, dv2._data_inst]).drop_duplicates()
        dv._factor_df = pd.concat([dv1._factor_df, dv2._factor_df])

        if not dv._factor_df.empty:
            dv._factor_df.drop_duplicates('factor_id')
            for index, row in dv._factor_df.iterrows():
                factor_id = row['factor_id']
                factor_body = row['factor_def']
                factor_args = list(filter(None, row['factor_args'].split(',')))
                if 'factor_quarterly' in row:
                    factor_quarterly = row['factor_quarterly']
                else:
                    factor_quarterly = False

                dv._import_factors[factor_id] = FactorDef(factor_id, factor_args, factor_body, factor_quarterly)

        dv.index_weights = {}
        for k,v in dv1.index_weights.items():
            if k in dv2.index_weights:
                df1 = v
                df2 = dv2.index_weights[k]
                dv.index_weights[k] = pd.concat( [df1[df1.index < dv2.start_date], df2[df2.index > dv1.end_date]] )

        dv.industry_groups = {}
        for k,v in dv1.industry_groups.items():
            if k in dv2.industry_groups:
                df1 = v
                df2 = dv2.industry_groups[k]
                dv.industry_groups[k] = pd.concat( [ df1[df1.index < dv2.start_date], df2[df2.index > dv1.end_date] ] )

        dv.data_d = concat_df(dv1, dv2, 'data_d')
        dv.data_q = concat_df(dv1, dv2, 'data_q')
        dv._data_benchmark = concat_df(dv1, dv2, '_data_benchmark')

        dv._process_data(True if dv1._snapshot else False)
        return dv

    def group_demean(self, signal, group, new_name, is_quarterly=False, method='div'):

        df = self.get_ts(signal + ',' + group + ',index_member')
        df_all = df.stack(level=0).reset_index()

        ## use only index member for demean
        df_all_mask = df_all[df_all['index_member'].astype(bool)]
        if method == 'div':
            df_all_mask[new_name] = df_all_mask.groupby(['trade_date', group])[signal].apply(lambda x: x / np.nanmedian(x))
        else:
            df_all_mask[new_name] = df_all_mask.groupby(['tpdrade_date', group])[signal].apply(lambda x: x - np.nanmedian(x))

        ## convert back to long data
        df_all_clean = df_all.loc[:, ['trade_date', 'symbol']]\
            .merge(df_all_mask, on=['trade_date', 'symbol'],how='left')\
            .pivot_table(values=new_name,
                         index='trade_date',
                         columns='symbol',
                         fill_value=np.nan)

        self.remove_field(new_name)
        self.append_df(df_all_clean, new_name, is_quarterly=is_quarterly)
        return self


    def to_dataframe(self):
        df = self.data_d.copy()
        df.columns = df.columns.swaplevel()
        return df.stack().reset_index()

    @staticmethod
    def from_dataframe(df):
        df = df.sort_values(['trade_date','symbol'])
        df = df.set_index(['trade_date','symbol']).unstack()
        df.columns = df.columns.swaplevel()
        df = df.sort_index(axis=1)

        dv = DataView()
        dv.data_d = df
        for field in df.columns.levels[1]:
            dv.fields.append(field)
        dv.start_date = df.index.min()
        dv.end_date = df.index.max()
        return dv


class EventDataView(object):
    """
    Prepare data before research / trade. Support file I/O.
    Support: add field, add formula, save / load.

    Attributes
    ----------
    data_api : RemoteDataService
    symbol : list
    start_date : int
    end_date : int
    fields : list
    freq : int
    market_daily_fields, reference_daily_fields : list
    data_d : pd.DataFrame
        All daily frequency data will be merged and stored here.
        index is date, columns is symbol-field MultiIndex
    data_q : pd.DataFrame
        All quarterly frequency data will be merged and stored here.
        index is date, columns is symbol-field MultiIndex

    """

    def __init__(self):
        self.data_api = None

        self.universe = ""
        self.symbol = []
        self.benchmark = ""
        self.start_date = 0
        self.extended_start_date_d = 0
        self.end_date = 0
        self.fields = []
        self.freq = 1
        self.all_price = True

        self.meta_data_list = ['start_date', 'end_date',
                               'extended_start_date_d',
                               'freq', 'fields', 'symbol', 'universe', 'benchmark',
                               'custom_daily_fields']
        self.adjust_mode = 'post'

        self.data_d = None
        self.data_q = None
        self._data_benchmark = None
        self._data_inst = None
        self.data_custom = None
        # self._data_group = None

        common_list = {'symbol', 'start_date', 'end_date'}
        market_bar_list = {'open', 'high', 'low', 'close', 'volume', 'turnover', 'vwap', 'oi'}
        market_tick_list = {'volume', 'oi',
                            'askprice1', 'askprice2', 'askprice3', 'askprice4', 'askprice5',
                            'bidprice1', 'bidprice1', 'bidprice1', 'bidprice1', 'bidprice1',
                            'askvolume1', 'askvolume2', 'askvolume3', 'askvolume4', 'askvolume5',
                            'bidvolume1', 'bidvolume2', 'bidvolume3', 'bidvolume4', 'bidvolume5'}
        # fields map
        # TODO: 'freq' is not in market_daily_fields yet.
        self.market_daily_fields = \
            {'open', 'high', 'low', 'close', 'volume', 'turnover', 'vwap', 'oi', 'trade_status',
             'open_adj', 'high_adj', 'low_adj', 'close_adj', 'vwap_adj', 'index_member', 'index_weight'}
        self.group_fields = {'sw1', 'sw2', 'sw3', 'sw4', 'zz1', 'zz2'}
        self.custom_daily_fields = []

        # const
        self.TRADE_STATUS_FIELD_NAME = 'trade_status'
        self.TRADE_DATE_FIELD_NAME = 'trade_date'

    # --------------------------------------------------------------------------------------------------------
    # Properties
    @property
    def data_benchmark(self):
        return self._data_benchmark

    @property
    def data_inst(self):
        """

        Returns
        -------
        pd.DataFrame

        """
        return self._data_inst

    @data_benchmark.setter
    def data_benchmark(self, df_new):
        if self.data_d is not None and df_new.shape[0] != self.data_d.shape[0]:
            raise ValueError("You must provide a DataFrame with the same shape of data_benchmark.")
        self._data_benchmark = df_new

    @property
    def dates(self):
        """
        Get daily date array of the underlying data.

        Returns
        -------
        res : np.array
            dtype: int

        """
        if self.data_d is not None:
            res = self.data_d.index.values
        elif self.data_api is not None:
            res = self.data_api.query_trade_dates(self.extended_start_date_d, self.end_date)
        else:
            raise ValueError("Cannot get dates array when neither of data and data_api exists.")

        return res

    # --------------------------------------------------------------------------------------------------------
    # Fields
    def _is_quarter_field(self, field_name):
        """
        Check whether a field name is quarterly frequency.
        field_name must be pre-defined or already added.

        Parameters
        ----------
        field_name : str

        Returns
        -------
        bool

        """
        res = (field_name in self.fin_stat_balance_sheet
               or field_name in self.fin_stat_cash_flow
               or field_name in self.fin_stat_income
               or field_name in self.fin_indicator
               or field_name in self.custom_quarterly_fields)
        return res

    def _is_daily_field(self, field_name):
        """
        Check whether a field name is daily frequency.
        field_name must be pre-defined or already added.

        Parameters
        ----------
        field_name : str

        Returns
        -------
        bool

        """
        flag = (field_name in self.market_daily_fields
                or field_name in self.reference_daily_fields
                or field_name in self.custom_daily_fields
                or field_name in self.group_fields)
        return flag

    def _is_predefined_field(self, field_name):
        """
        Check whether a field name can be recognized.
        field_name must be pre-defined or already added.

        Parameters
        ----------
        field_name : str

        Returns
        -------
        bool

        """
        return self._is_quarter_field(field_name) or self._is_daily_field(field_name)

    def _get_fields(self, field_type, fields, complement=False, append=False):
        """
        Get list of fields that are in ref_quarterly_fields.

        Parameters
        ----------
        field_type : {'market_daily', 'ref_daily', 'income', 'balance_sheet', 'cash_flow', 'daily', 'quarterly'
        fields : list of str
        complement : bool, optional
            If True, get fields that are NOT in ref_quarterly_fields.

        Returns
        -------
        list

        """
        pool_map = {'market_daily': self.market_daily_fields,
                    'group': self.group_fields}
        pool_map['daily'] = set.union(pool_map['market_daily'],
                                      pool_map['group'],
                                      self.custom_daily_fields)

        pool = pool_map.get(field_type, None)
        if pool is None:
            raise NotImplementedError("field_type = {:s}".format(field_type))

        s = set.intersection(set(pool), set(fields))
        if not s:
            return []

        if complement:
            s = set(fields) - s

        if field_type == 'market_daily' and self.all_price:
            # turnover will not be adjusted
            s.update({'open', 'high', 'close', 'low', 'vwap'})

        if append:
            s.add('symbol')
            if field_type == 'market_daily' or field_type == 'ref_daily':
                s.add('trade_date')
                if field_type == 'market_daily':
                    s.add(self.TRADE_STATUS_FIELD_NAME)
            elif (field_type == 'income'
                  or field_type == 'balance_sheet'
                  or field_type == 'cash_flow'
                  or field_type == 'fin_indicator'):
                s.add(self.ANN_DATE_FIELD_NAME)
                s.add(self.REPORT_DATE_FIELD_NAME)

        l = list(s)
        return l

    # --------------------------------------------------------------------------------------------------------
    # Prepare data
    def init_from_config(self, props, data_api):
        """
        Initialize various attributes like start/end date, universe/symbol, fields, etc.
        If your want to parse symbol, but use a custom benchmark index, please directly assign self.data_benchmark.

        Parameters
        ----------
        props : dict
            start_date, end_date, freq, symbol, fields, etc.
        data_api : BaseDataServer

        """
        # data_api.init_from_config(props)
        self.data_api = data_api

        sep = ','

        # initialize parameters
        self.start_date = props['start_date']
        self.extended_start_date_d = jutil.shift(self.start_date, n_weeks=-8)  # query more data
        self.extended_start_date_q = jutil.shift(self.start_date, n_weeks=-130)
        self.end_date = props['end_date']
        self.all_price = props.get('all_price', True)
        self.freq = props.get('freq', 1)

        # get and filter fields
        fields = props.get('fields', [])
        if fields:
            fields = props['fields'].split(sep)
            self.fields = [field for field in fields if self._is_predefined_field(field)]
            if len(self.fields) < len(fields):
                print("Field name [{}] not valid, ignore.".format(set.difference(set(fields), set(self.fields))))

        # append additional fields
        if self.all_price:
            self.fields.extend(['open_adj', 'high_adj', 'low_adj', 'close_adj',
                                'open', 'high', 'low', 'close',
                                'vwap', 'vwap_adj'])

        # always inclde sw1 for industry analysis
        if 'sw1' not in self.fields:
            self.fields.extend(['sw1'])

        # initialize universe/symbol
        universe = props.get('universe', "")
        symbol = props.get('symbol', "")
        benchmark = props.get('benchmark', '')
        if symbol and universe:
            raise ValueError("Please use either [symbol] or [universe].")
        if not (symbol or universe):
            raise ValueError("One of [symbol] or [universe] must be provided.")
        if universe:
            univ_list = universe.split(',')
            self.universe = univ_list
            symbols_list = []
            for univ in univ_list:
                symbols_list.extend(data_api.query_index_member(univ, self.extended_start_date_d, self.end_date))
            self.symbol = sorted(list(set(symbols_list)))
        else:
            self.symbol = sorted(symbol.split(sep))
        if benchmark:
            self.benchmark = benchmark
        else:
            if self.universe:
                if len(self.universe) > 1:
                    print("More than one universe are used: {}, "
                          "use the first one ({}) as index by default. "
                          "If you want to use other benchmark, "
                          "please specify benchmark in configs.".format(repr(self.universe), self.universe[0]))
                self.benchmark = self.universe[0]

        print("Initialize config success.")

    def distributed_query(self, query_func_name, symbol, start_date, end_date, limit=100000, **kwargs):
        n_symbols = len(symbol.split(','))
        dates = self.data_api.query_trade_dates(start_date, end_date)
        n_days = len(dates)

        if n_symbols * n_days > limit:
            n = limit // n_symbols

            df_list = []
            i = 0
            pos1, pos2 = n * i, n * (i + 1) - 1
            while pos2 < n_days:
                print(pos2)
                df, msg = getattr(self.data_api, query_func_name)(symbol=symbol,
                                                                  start_date=dates[pos1], end_date=dates[pos2],
                                                                  **kwargs)
                df_list.append(df)
                i += 1
                pos1, pos2 = n * i, n * (i + 1) - 1
            if pos1 < n_days:
                df, msg = getattr(self.data_api, query_func_name)(symbol=symbol,
                                                                  start_date=dates[pos1], end_date=dates[-1],
                                                                  **kwargs)
                df_list.append(df)
            df = pd.concat(df_list, axis=0)
        else:
            df, msg = getattr(self.data_api, query_func_name)(symbol,
                                                              start_date=start_date, end_date=end_date,
                                                              **kwargs)
        return df, msg

    def prepare_data(self):
        """Prepare data for the FIRST time."""
        # prepare benchmark and group
        print("Query data...")
        data_d = self._prepare_daily_quarterly(self.fields)
        self.data_d = data_d
        # self._align_and_merge_q_into_d()

        print("Query instrument info...")
        self._prepare_inst_info()

        if self.benchmark:
            print("Query benchmark...")
            self._data_benchmark = self._prepare_benchmark()

        print("Data has been successfully prepared.")

    @staticmethod
    def _process_index_co(df, index_name):
        df = df.astype(dtype={index_name: int})
        df = df.drop_duplicates(subset=['symbol', index_name])
        return df

    def _prepare_daily_quarterly(self, fields):
        """
        Query and process data from data_api.

        Parameters
        ----------
        fields : list

        Returns
        -------
        merge_d : pd.DataFrame or None
        merge_q : pd.DataFrame or None

        """
        if not fields:
            return None, None

        # query data
        print("Query data - query...")
        daily_list = self._query_data(self.symbol, fields)

        def pivot_and_sort(df, index_name):
            df = self._process_index_co(df, index_name)
            df = df.pivot(index=index_name, columns='symbol')
            df.columns = df.columns.swaplevel()
            col_names = ['symbol', 'field']
            df.columns.names = col_names
            df = df.sort_index(axis=1, level=col_names)
            df.index.name = index_name
            return df

        multi_daily = None
        if daily_list:
            daily_list_pivot = [pivot_and_sort(df, self.TRADE_DATE_FIELD_NAME) for df in daily_list]
            multi_daily = self._merge_data(daily_list_pivot, self.TRADE_DATE_FIELD_NAME)
            # use self.dates as index because original data have weekends
            multi_daily = self._fill_missing_idx_col(multi_daily, index=self.dates, symbols=self.symbol)
            print("Query data - daily fields prepared.")

        return multi_daily

    def _query_data(self, symbol, fields):
        """
        Query data using different APIs, then store them in dict.
        period, start_date and end_date are fixed.
        Keys of dict are securitites.

        Parameters
        ----------
        symbol : list of str
        fields : list of str

        Returns
        -------
        daily_list : list
        quarterly_list : list

        """
        sep = ','
        symbol_str = sep.join(symbol)

        if self.freq == 1:
            daily_list = []

            # TODO : use fields = {field: kwargs} to enable params
            fields_market_daily = self._get_fields('market_daily', fields, append=True)
            if fields_market_daily:
                print("NOTE: price adjust method is [{:s} adjust]".format(self.adjust_mode))
                # no adjust prices and other market daily fields
                df_daily, msg1 = self.distributed_query('daily', symbol_str,
                                                        start_date=self.extended_start_date_d, end_date=self.end_date,
                                                        adjust_mode=None, fields=sep.join(fields_market_daily),
                                                        limit=100000)
                # df_daily, msg1 = self.data_api.daily(symbol_str, start_date=self.extended_start_date_d, end_date=self.end_date,
                #                                     adjust_mode=None, fields=sep.join(fields_market_daily))

                if self.all_price:
                    adj_cols = ['open', 'high', 'low', 'close', 'vwap']
                    # adjusted prices
                    # df_daily_adjust, msg11 = self.data_api.daily(symbol_str, start_date=self.extended_start_date_d, end_date=self.end_date,
                    #                                             adjust_mode=self.adjust_mode, fields=','.join(adj_cols))
                    df_daily_adjust, msg1 = self.distributed_query('daily', symbol_str,
                                                                   start_date=self.extended_start_date_d,
                                                                   end_date=self.end_date,
                                                                   adjust_mode=self.adjust_mode,
                                                                   fields=sep.join(fields_market_daily), limit=100000)

                    df_daily = pd.merge(df_daily, df_daily_adjust, how='outer',
                                        on=['symbol', 'trade_date'], suffixes=('', '_adj'))
                daily_list.append(df_daily.loc[:, fields_market_daily])

        else:
            raise NotImplementedError("freq = {}".format(self.freq))

        return daily_list

    @staticmethod
    def _merge_data(dfs, index_name='trade_date'):
        """
        Merge data from different APIs into one DataFrame.

        Parameters
        ----------
        dfs : list of pd.DataFrame

        Returns
        -------
        merge : pd.DataFrame or None
            If dfs is empty, return None

        Notes
        -----
        Align on date index, concatenate on columns (symbol and fields)

        """
        # dfs = [df for df in dfs if df is not None]

        merge = pd.concat(dfs, axis=1, join='outer')

        # drop duplicated columns. ONE LINE EFFICIENT version
        mask_duplicated = merge.columns.duplicated()
        if np.any(mask_duplicated):
            # print("Duplicated columns found. Dropped.")
            merge = merge.loc[:, ~mask_duplicated]

            # if merge.isnull().sum().sum() > 0:
            # print "WARNING: nan in final merged data. NO fill"
            # merge.fillna(method='ffill', inplace=True)

        merge = merge.sort_index(axis=1, level=['symbol', 'field'])
        merge.index.name = index_name

        return merge

    def _fill_missing_idx_col(self, df, index=None, symbols=None):
        if index is None:
            index = df.index
        if symbols is None:
            symbols = self.symbol
        fields = df.columns.levels[1]

        if len(fields) * len(self.symbol) != len(df.columns) or len(index) != len(df.index):
            cols_multi = pd.MultiIndex.from_product([symbols, fields], names=['symbol', 'field'])
            cols_multi = cols_multi.sort_values()
            df_final = pd.DataFrame(index=index, columns=cols_multi, data=np.nan)
            df_final.index.name = df.index.name

            df_final.update(df)

            # idx_diff = sorted(set(df_final.index) - set(df.index))
            col_diff = sorted(set(df_final.columns.levels[0].values) - set(df.columns.levels[0].values))
            print("WARNING: some data is unavailable: "
                  # + "\n    At index " + ', '.join(idx_diff)
                  + "\n    At fields " + ', '.join(col_diff))
            return df_final
        else:
            return df

    def _prepare_adj_factor(self):
        """Query and append daily adjust factor for prices."""
        mask_stocks = self.data_inst['inst_type'] == 1
        if mask_stocks.sum() == 0:
            return
        symbol_stocks = self.data_inst.loc[mask_stocks].index.values
        symbol_str = ','.join(symbol_stocks)
        df_adj = self.data_api.query_adj_factor_daily(symbol_str,
                                                      start_date=self.extended_start_date_d, end_date=self.end_date,
                                                      div=False)
        self.append_df(df_adj, 'adjust_factor', is_quarterly=False)

    def _prepare_comp_info(self):
        # if a symbol is index member of any one universe, its value of index_member will be 1.0
        res = dict()
        for univ in self.universe:
            df = self.data_api.query_index_member_daily(univ, self.extended_start_date_d, self.end_date)
            res[univ] = df
        df_res = pd.concat(res, axis=0)
        df = df_res.groupby(by='trade_date').apply(lambda df: df.any(axis=0)).astype(float)
        self.append_df(df, 'index_member', is_quarterly=False)

        # use weights of the first universe
        df_weights = self.data_api.query_index_weights_daily(self.universe[0], self.extended_start_date_d,
                                                             self.end_date)
        self.append_df(df_weights, 'index_weight', is_quarterly=False)

    def _prepare_inst_info(self):
        res = self.data_api.query_inst_info(symbol=','.join(self.symbol),
                                            fields='symbol,inst_type,name,list_date,'
                                                   'delist_date,product,pricetick,multiplier,'
                                                   'buylot,setlot',
                                            inst_type="")
        self._data_inst = res

    def _prepare_group(self, group_fields):
        data_map = {'sw1': ('SW', 1),
                    'sw2': ('SW', 2),
                    'sw3': ('SW', 3),
                    'sw4': ('SW', 4),
                    'zz1': ('ZZ', 1),
                    'zz2': ('ZZ', 2)}
        for field in group_fields:
            type_, level = data_map[field]
            df = self.data_api.query_industry_daily(symbol=','.join(self.symbol),
                                                    start_date=self.extended_start_date_q, end_date=self.end_date,
                                                    type_=type_, level=level)
            self.append_df(df, field, is_quarterly=False)

    def _prepare_benchmark(self):
        df_bench, msg = self.data_api.daily(self.benchmark,
                                            start_date=self.extended_start_date_d, end_date=self.end_date,
                                            adjust_mode=self.adjust_mode,
                                            fields='trade_date,symbol,close,vwap,volume,turnover')
        # TODO: we want more than just close price of benchmark
        df_bench = df_bench.set_index('trade_date').loc[:, ['close']]
        return df_bench

    # --------------------------------------------------------------------------------------------------------
    # Add/Remove Fields&Formulas
    def _add_field(self, field_name, is_quarterly=None):
        self.fields.append(field_name)
        if not self._is_predefined_field(field_name):
            self.custom_daily_fields.append(field_name)

    def _add_symbol(self, symbol_name):
        if symbol_name in self.symbol:
            print("symbol [{:s}] already exists, add_symbol failed.".format(symbol_name))
            return
        self.symbol.append(symbol_name)

    def add_field(self, field_name, data_api=None):
        """
        Query and append new field to DataView.

        Parameters
        ----------
        data_api : BaseDataServer
        field_name : str
            Must be a known field name (which is given in documents).

        Returns
        -------
        bool
            whether add successfully.

        """
        if data_api is None:
            if self.data_api is None:
                print("Add field failed. No data_api available. Please specify one in parameter.")
                return False
        else:
            self.data_api = data_api

        if field_name in self.fields:
            print("Field name [{:s}] already exists.".format(field_name))
            return False

        if not self._is_predefined_field(field_name):
            print("Field name [{}] not valid, ignore.".format(field_name))
            return False

        merge_d, merge_q = self._prepare_daily_quarterly([field_name])

        if self._is_daily_field(field_name):
            if self.data_d is None:
                raise ValueError("Please prepare [{:s}] first.".format(field_name))
            merge, _ = self._prepare_daily_quarterly([field_name])
            is_quarterly = False
        else:
            if self.data_q is None:
                raise ValueError("Please prepare [{:s}] first.".format(field_name))
            _, merge = self._prepare_daily_quarterly([field_name])
            is_quarterly = True

        merge = merge.loc[:, pd.IndexSlice[:, field_name]]
        merge.columns = merge.columns.droplevel(level=1)
        self.append_df(merge, field_name,
                       is_quarterly=is_quarterly)  # whether contain only trade days is decided by existing data.

        if is_quarterly:
            df_ann = merge_q.loc[:, pd.IndexSlice[:, self.ANN_DATE_FIELD_NAME]]
            df_ann.columns = df_ann.columns.droplevel(level='field')
            df_expanded = align(merge, df_ann, self.dates)
            self.append_df(df_expanded, field_name, is_quarterly=False)
        return True

    def add_formula(self, field_name, formula, is_quarterly, overwrite=True,
                    formula_func_name_style='camel', data_api=None,
                    within_index=True, is_factor=True):
        """
        Add a new field, which is calculated using existing fields.

        Parameters
        ----------
        formula : str
            A formula contains operations and function calls.
        field_name : str
            A custom name for the new field.
        is_quarterly : bool
            Whether df is quarterly data (like quarterly financial statement) or daily data.
        overwrite : bool, optional
            Whether overwrite existing field. True by default.
        formula_func_name_style : {'upper', 'lower'}, optional
        data_api : RemoteDataService, optional
        within_index : bool
            When do cross-section operatioins, whether just do within index components.

        Notes
        -----
        Time cost of this function:
            For a simple formula (like 'a + 1'), almost all time is consumed by append_df;
            For a complex formula (like 'GroupRank'), half of time is consumed by evaluation and half by append_df.
        """
        if data_api is not None:
            self.data_api = data_api

        if field_name in self.fields:
            if overwrite:
                self.remove_field(field_name)
                print("Field [{:s}] is overwritten.".format(field_name))
            else:
                print("Add formula failed: name [{:s}] exist. Try another name.".format(field_name))
                return

        parser = Parser()
        parser.set_capital(formula_func_name_style)

        expr = parser.parse(formula)

        var_df_dic = dict()
        var_list = expr.variables()

        # TODO: users do not need to prepare data before add_formula
        if not self.fields:
            self.fields.extend(var_list)
            self.prepare_data()
        else:
            for var in var_list:
                if var not in self.fields:
                    print("Variable [{:s}] is not recognized (it may be wrong)," \
                          "try to fetch from the server...".format(var))
                    success = self.add_field(var)
                    if not success:
                        return

        for var in var_list:
            if self._is_quarter_field(var):
                df_var = self.get_ts_quarter(var, start_date=self.extended_start_date_q)
            else:
                # must use extended date. Default is start_date
                df_var = self.get_ts(var, start_date=self.extended_start_date_d, end_date=self.end_date)

            var_df_dic[var] = df_var

        # TODO: send ann_date into expr.evaluate. We assume that ann_date of all fields of a symbol is the same
        df_ann = self._get_ann_df()
        if within_index:
            df_index_member = self.get_ts('index_member', start_date=self.extended_start_date_d, end_date=self.end_date)
            df_eval = parser.evaluate(var_df_dic, ann_dts=df_ann, trade_dts=self.dates, index_member=df_index_member)
        else:
            df_eval = parser.evaluate(var_df_dic, ann_dts=df_ann, trade_dts=self.dates)

        self.append_df(df_eval, field_name, is_quarterly=is_quarterly)

        if is_quarterly:
            df_ann = self._get_ann_df()
            df_expanded = align(df_eval, df_ann, self.dates)
            self.append_df(df_expanded, field_name, is_quarterly=False)

    def append_df(self, df, field_name, is_quarterly=False):
        """
        Append DataFrame to existing multi-index DataFrame and add corresponding field name.

        Parameters
        ----------
        df : pd.DataFrame or pd.Series
        field_name : str
        is_quarterly : bool
            Whether df is quarterly data (like quarterly financial statement) or daily data.

        Notes
        -----
        append_df does not support overwrite. To overwrite a field, you must first do self.remove_fields(),
        then append_df() again.

        """
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            pass
        elif isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        else:
            raise ValueError("Data to be appended must be pandas format. But we have {}".format(type(df)))

        if is_quarterly:
            the_data = self.data_q
        else:
            the_data = self.data_d

        exist_symbols = the_data.columns.levels[0]
        if len(df.columns) < len(exist_symbols):
            df2 = pd.DataFrame(index=df.index, columns=exist_symbols, data=np.nan)
            df2.update(df)
            df = df2
        elif len(df.columns) > len(exist_symbols):
            df = df.loc[:, exist_symbols]
        multi_idx = pd.MultiIndex.from_product([exist_symbols, [field_name]])
        df.columns = multi_idx

        # the_data = apply_in_subprocess(pd.merge, args=(the_data, df),
        #                            kwargs={'left_index': True, 'right_index': True, 'how': 'left'})  # runs in *only* one process
        the_data = pd.merge(the_data, df, left_index=True, right_index=True, how='left')
        the_data = the_data.sort_index(axis=1)
        # merge = the_data.join(df, how='left')  # left: keep index of existing data unchanged
        # sort_columns(the_data)

        if is_quarterly:
            self.data_q = the_data
        else:
            self.data_d = the_data
        self._add_field(field_name, is_quarterly)

    def append_df_symbol(self, df, symbol_name):
        """
        Append DataFrame to existing multi-index DataFrame and add corresponding field name.

        Parameters
        ----------
        df : pd.DataFrame or pd.Series
        symbol_name : str
        is_quarterly : bool
            Whether df is quarterly data (like quarterly financial statement) or daily data.

        Notes
        -----
        append_df does not support overwrite. To overwrite a field, you must first do self.remove_fields(),
        then append_df() again.

        """
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            pass
        elif isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        else:
            raise ValueError("Data to be appended must be pandas format. But we have {}".format(type(df)))

        the_data = self.data_d

        exist_fields = the_data.columns.levels[1]
        if len(set(exist_fields) - set(df.columns)):
            # if set(df.columns) < set(exist_fields):
            df2 = pd.DataFrame(index=df.index, columns=exist_fields, data=np.nan)
            df2.update(df)
            df = df2
        multi_idx = pd.MultiIndex.from_product([[symbol_name], exist_fields])
        df.columns = multi_idx

        # the_data = apply_in_subprocess(pd.merge, args=(the_data, df),
        #                            kwargs={'left_index': True, 'right_index': True, 'how': 'left'})  # runs in *only* one process
        the_data = pd.merge(the_data, df, left_index=True, right_index=True, how='left')
        the_data = the_data.sort_index(axis=1)
        # merge = the_data.join(df, how='left')  # left: keep index of existing data unchanged
        # sort_columns(the_data)

        self.data_d = the_data
        self._add_symbol(symbol_name)

    def remove_field(self, field_names):
        """
        Query and append new field to DataView.

        Parameters
        ----------
        field_names : str or list
            The (custom) field to be removed from dataview.

        Returns
        -------
        bool
            whether add successfully.

        """
        if isinstance(field_names, basestring):
            field_names = field_names.split(',')
        elif isinstance(field_names, (list, tuple)):
            pass
        else:
            raise ValueError("field_names must be str or list of str.")

        for field_name in field_names:
            # parameter validation
            if field_name not in self.fields:
                print("Field name [{:s}] does not exist.".format(field_name))
                return

            if self._is_daily_field(field_name):
                is_quarterly = False
            elif self._is_quarter_field(field_name):
                is_quarterly = True
            else:
                print("Field name [{}] is a pre-defined field, ignore.".format(field_name))
                return

            # remove field data

            self.data_d = self.data_d.drop(field_name, axis=1, level=1)
            if is_quarterly:
                self.data_q = self.data_q.drop(field_name, axis=1, level=1)

            # remove fields name from list
            self.fields.remove(field_name)
            if is_quarterly:
                if field_name in self.custom_quarterly_fields:
                    self.custom_quarterly_fields.remove(field_name)
            else:
                if field_name in self.custom_daily_fields:
                    self.custom_daily_fields.remove(field_name)

    # --------------------------------------------------------------------------------------------------------
    # Get Data API
    def get(self, symbol="", start_date=0, end_date=0, fields="", data_format='wide'):
        """
        Basic API to get arbitrary data. If nothing fetched, return None.

        Parameters
        ----------
        symbol : str, optional
            Separated by ',' default "" (all securities).
        start_date : int, optional
            Default 0 (self.start_date).
        end_date : int, optional
            Default 0 (self.start_date).
        fields : str, optional
            Separated by ',' default "" (all fields).
        data_format : {'long', 'wide'}, optional
            Format of result DataFrame, default 'wide'.

        Returns
        -------
        res : pd.DataFrame or None
            index is datetimeindex, columns are (symbol, fields) MultiIndex

        """
        sep = ','

        if not fields:
            fields = slice(None)  # self.fields
        else:
            fields = fields.split(sep)

        if not symbol:
            symbol = slice(None)  # this is 3X faster than symbol = self.symbol
        else:
            symbol = symbol.split(sep)

        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date

        res = self.data_d.loc[pd.IndexSlice[start_date: end_date], pd.IndexSlice[symbol, fields]]

        if data_format == 'wide':
            pass
        else:
            res = res.stack(level='symbol').reset_index()
        return res

    def get_snapshot(self, snapshot_date, symbol="", fields=""):
        """
        Get snapshot of given fields and symbol at snapshot_date.

        Parameters
        ----------
        snapshot_date : int
            Date of snapshot.
        symbol : str, optional
            Separated by ',' default "" (all securities).
        fields : str, optional
            Separated by ',' default "" (all fields).

        Returns
        -------
        res : pd.DataFrame
            symbol as index, field as columns

        """
        res = self.get(symbol=symbol, start_date=snapshot_date, end_date=snapshot_date, fields=fields)
        if res is None:
            print("No data. for date={}, fields={}, symbol={}".format(snapshot_date, fields, symbol))
            return

        res = res.stack(level='symbol', dropna=False)
        res.index = res.index.droplevel(level=self.TRADE_DATE_FIELD_NAME)

        return res

    def _get_ann_df(self):
        """
        Query announcement date of financial statements of all securities.

        Returns
        -------
        df_ann : pd.DataFrame or None
            Index is date, column is symbol.
            If no quarterly data available, return None.

        """
        if self.data_q is None:
            return None
        df_ann = self.data_q.loc[:, pd.IndexSlice[:, self.ANN_DATE_FIELD_NAME]]
        df_ann.columns = df_ann.columns.droplevel(level='field')

        return df_ann

    def get_symbol(self, symbol, start_date=0, end_date=0, fields=""):
        res = self.get(symbol, start_date=start_date, end_date=end_date, fields=fields)
        if res is None:
            raise ValueError("No data. for "
                             "start_date={}, end_date={}, field={}, symbol={}".format(start_date, end_date,
                                                                                      fields, symbol))

        res.columns = res.columns.droplevel(level='symbol')
        return res

    def get_ts(self, field, symbol="", start_date=0, end_date=0):
        """
        Get time series data of single field.

        Parameters
        ----------
        field : str
            Single field.
        symbol : str, optional
            Separated by ',' default "" (all securities).
        start_date : int, optional
            Default 0 (self.start_date).
        end_date : int, optional
            Default 0 (self.start_date).

        Returns
        -------
        res : pd.DataFrame
            Index is int date, column is symbol.

        """
        res = self.get(symbol, start_date=start_date, end_date=end_date, fields=field)
        if res is None:
            raise ValueError("No data. for "
                             "start_date={}, end_date={}, field={}, symbol={}".format(start_date, end_date,
                                                                                      field, symbol))

        res.columns = res.columns.droplevel(level='field')
        return res

    # --------------------------------------------------------------------------------------------------------
    # DataView I/O
    @staticmethod
    def _load_h5(fp):
        """Load data and meta_data from hd5 file.

        Parameters
        ----------
        fp : str, optional
            File path of pre-stored hd5 file.

        """
        h5 = pd.HDFStore(fp)

        res = dict()
        for key in h5.keys():
            res[key] = h5.get(key)

        h5.close()

        return res

    def load_dataview(self, folder_path='.'):
        """
        Load data from local file.

        Parameters
        ----------
        folder_path : str, optional
            Folder path to store hd5 file and meta data.

        """
        path_meta_data = os.path.join(folder_path, 'meta_data.json')
        path_data = os.path.join(folder_path, 'data.hd5')
        if not (os.path.exists(path_meta_data) and os.path.exists(path_data)):
            raise IOError("There is no data file under directory {}".format(folder_path))

        meta_data = jutil.read_json(path_meta_data)
        dic = self._load_h5(path_data)
        self.data_d = dic.get('/data_d', None)
        self.data_q = dic.get('/data_q', None)
        self._data_benchmark = dic.get('/data_benchmark', None)
        self._data_inst = dic.get('/data_inst', None)
        self.data_custom = dic.get('/data_custom', None)
        self.__dict__.update(meta_data)

        print("Dataview loaded successfully.")

    def save_dataview(self, folder_path):
        """
        Save data and meta_data_to_store to a single hd5 file.
        Store at output/sub_folder

        Parameters
        ----------
        folder_path : str
            Path to store your data.

        """
        abs_folder = os.path.abspath(folder_path)
        meta_path = os.path.join(folder_path, 'meta_data.json')
        data_path = os.path.join(folder_path, 'data.hd5')

        data_to_store = {'data_d': self.data_d, 'data_q': self.data_q,
                         'data_benchmark': self.data_benchmark, 'data_inst': self.data_inst,
                         'data_custom': self.data_custom}
        data_to_store = {k: v for k, v in data_to_store.items() if v is not None}
        meta_data_to_store = {key: self.__dict__[key] for key in self.meta_data_list}

        print("\nStore data...")
        jutil.save_json(meta_data_to_store, meta_path)
        self._save_h5(data_path, data_to_store)

        print("Dataview has been successfully saved to:\n"
              + abs_folder + "\n\n"
              + "You can load it with load_dataview('{:s}')".format(abs_folder))

    @staticmethod
    def _save_h5(fp, dic):
        """
        Save data in dic to a hd5 file.

        Parameters
        ----------
        fp : str
            File path.
        dic : dict

        """
        import warnings
        warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

        jutil.create_dir(fp)
        h5 = pd.HDFStore(fp, complevel=9, complib='blosc')
        for key, value in dic.items():
            h5[key] = value
        h5.close()


    # --------------------------------------------------------------------------------------------------------
    # DataView I/O
