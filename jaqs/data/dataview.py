# encoding: utf-8
"""
How to add custom (alternative) data:
    1. Build a DataFrame of your data, whose index is dv.dates, column is dv.symbol.
    2. Use dv.append_df to add your DataFrame to the DataView object.
    
If you will use this data frequently, you can add define a new method of DataServer, then get data from your DataServer.
If you want to declare your field in props, instead of append it manually, you will have to modify prepare_data function.
"""
import os

import numpy as np
import pandas as pd

import jaqs.util.fileio
from jaqs.util import dtutil
from jaqs.data.align import align
from jaqs.data.py_expression_eval import Parser


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
    # TODO only support stocks!
    def __init__(self):
        self.data_api = None
        
        self.universe = ""
        self.symbol = []
        self.start_date = 0
        self.extended_start_date_d = 0
        self.extended_start_date_q = 0
        self.end_date = 0
        self.fields = []
        self.freq = 1
        self.all_price = True

        self.meta_data_list = ['start_date', 'end_date',
                               'extended_start_date_d', 'extended_start_date_q',
                               'freq', 'fields', 'symbol', 'universe',
                               'custom_daily_fields', 'custom_quarterly_fields']
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
        self.market_daily_fields = \
            {'open', 'high', 'low', 'close', 'volume', 'turnover', 'vwap', 'oi', 'trade_status',
             'open_adj', 'high_adj', 'low_adj', 'close_adj', 'index_member'}
        self.group_fields = {'sw1', 'sw2', 'sw3', 'sw4', 'zz1', 'zz2'}
        self.reference_daily_fields = \
            {'currency', 'total_market_value', 'float_market_value',
             'high_52w', 'low_52w', 'high_52w_adj', 'low_52w_adj', 'close_price',
             'price_level', 'limit_status',
             'pe', 'pb', 'pe_ttm', 'pcf', 'pcf_ttm', 'ncf', 'ncf_ttm', 'ps', 'ps_ttm',
             'turnover_ratio', 'turnover_ratio_float',
             'share_amount', 'share_float', 'price_div_dps',
             'share_float_free', 'nppc_ttm', 'nppc_lyr', 'net_assets',
             'ncfoa_ttm', 'ncfoa_lyr', 'rev_ttm', 'rev_lyr', 'nicce_ttm'}
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
            {"extraordinary","deductedprofit","grossmargin","operateincome","investincome","stmnote_finexp",
             "stm_is","ebit","ebitda""fcff","fcfe","exinterestdebt_current","exinterestdebt_noncurrent","interestdebt",
             "netdebt","tangibleasset","workingcapital","networkingcapital","investcapital","retainedearnings","eps_basic_daily", # TODO eps_basic
             "eps_diluted","eps_diluted2","bps","ocfps","grps","orps","surpluscapitalps","surplusreserveps","undistributedps",
             "retainedps","cfps","ebitps","fcffps","fcfeps","netprofitmargin","grossprofitmargin","cogstosales",
             "expensetosales","profittogr","saleexpensetogr","adminexpensetogr","finaexpensetogr","impairtogr_ttm",
             "gctogr","optogr","ebittogr","roe","roe_deducted","roa2","roa","roic","roe_yearly","roa2_yearly","roe_avg",
             "operateincometoebt","investincometoebt","nonoperateprofittoebt","taxtoebt","deductedprofittoprofit","salescashintoor",
             "ocftoor","ocftooperateincome","capitalizedtoda","debttoassets","assetstoequity","dupont_assetstoequity",
             "catoassets","ncatoassets","tangibleassetstoassets","intdebttototalcap","equitytototalcapital","currentdebttodebt",
             "longdebtodebt","current","quick","cashratio","ocftoshortdebt","debttoequity","equitytodebt",
             "equitytointerestdebt","tangibleassettodebt","tangassettointdebt","tangibleassettonetdebt","ocftodebt",
             "ocftointerestdebt","ocftonetdebt","ebittointerest","longdebttoworkingcapital","ebitdatodebt","turndays",
             "invturndays","arturndays","invturn","arturn","caturn","faturn","assetsturn","roa_yearly","dupont_roa",
             "s_stm_bs","prefinexpense_opprofit","nonopprofit","optoebt","noptoebt","ocftoprofit","cashtoliqdebt",
             "cashtoliqdebtwithinterest","optoliqdebt","optodebt","roic_yearly","tot_faturn","profittoop","qfa_operateincome",
             "qfa_investincome","qfa_deductedprofit","qfa_eps","qfa_netprofitmargin","qfa_grossprofitmargin","qfa_expensetosales",
             "qfa_profittogr","qfa_saleexpensetogr","qfa_adminexpensetogr","qfa_finaexpensetogr","qfa_impairtogr_ttm",
             "qfa_gctogr","qfa_optogr","qfa_roe","qfa_roe_deducted","qfa_roa","qfa_operateincometoebt","qfa_investincometoebt",
             "qfa_deductedprofittoprofit","qfa_salescashintoor","qfa_ocftosales","qfa_ocftoor","yoyeps_basic","yoyeps_diluted",
             "yoyocfps","yoyop","yoyebt","yoynetprofit","yoynetprofit_deducted","yoyocf","yoyroe","yoybps","yoyassets",
             "yoyequity","yoy_tr","yoy_or","qfa_yoygr","qfa_cgrgr","qfa_yoysales","qfa_cgrsales","qfa_yoyop","qfa_cgrop",
             "qfa_yoyprofit","qfa_cgrprofit","qfa_yoynetprofit","qfa_cgrnetprofit","yoy_equity","rd_expense","waa_roe"}
        self .custom_daily_fields = []
        self .custom_quarterly_fields = []
        
        # co nst
        self .ANN_DATE_FIELD_NAME = 'ann_date'
        self .REPORT_DATE_FIELD_NAME = 'report_date'
        self.TRADE_STATUS_FIELD_NAME = 'trade_status'
        self.TRADE_DATE_FIELD_NAME = 'trade_date'
    
    @property
    def data_benchmark(self):
        return self._data_benchmark
    '''
    
    @property
    def data_group(self):
        """
        
        Returns
        -------
        pd.DataFrame

        """
        if self._data_group is None:
            self._data_group = self.get_ts('group')
        return self._data_group
    '''
    
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
        if self._data_benchmark.shape[0] != df_new.shape[0]:
            raise ValueError("You must provide a DataFrame with the same shape of data_benchmark.")
        self._data_benchmark = df_new
    
    @staticmethod
    def _group_df_to_dict(df, by):
        gp = df.groupby(by=by)
        res = {key: value for key, value in gp}
        return res

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
        if field_type == 'market_daily':
            pool = self.market_daily_fields
        elif field_type == 'ref_daily':
            pool = self.reference_daily_fields
        elif field_type == 'income':
            pool = self.fin_stat_income
        elif field_type == 'balance_sheet':
            pool = self.fin_stat_balance_sheet
        elif field_type == 'cash_flow':
            pool = self.fin_stat_cash_flow
        elif field_type == 'fin_indicator':
            pool = self.fin_indicator
        elif field_type == 'group':
            pool = self.group_fields
        elif field_type == 'daily':
            pool = set.union(self.market_daily_fields, self.reference_daily_fields,
                             self.custom_daily_fields, self.group_fields)
        elif field_type == 'quarterly':
            pool = set.union(self.fin_stat_income, self.fin_stat_balance_sheet, self.fin_stat_cash_flow,
                             self.fin_indicator,
                             self.custom_quarterly_fields)
        else:
            raise NotImplementedError("field_type = {:s}".format(field_type))
        
        s = set.intersection(set(pool), set(fields))
        if not s:
            return []
        
        if complement:
            s = set(fields) - s
            
        if field_type == 'market_daily':
            # turnover will not be adjusted
            s.update({'open', 'high', 'close', 'low', 'vwap'})
            
        if append:
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
        dic_market_daily : dict
            {str: DataFrame}
        dic_ref_daily : dict
            {str: DataFrame}
        dic_income : dict
            {str: DataFrame}
        dic_cash_flow : dict
            {str: DataFrame}
        dic_balance_sheet : dict
            {str: DataFrame}
        dic_fin_ind : dict
            {str: DataFrame}

        """
        sep = ','
        symbol_str = sep.join(symbol)
        
        dic_ref_daily = None
        dic_market_daily = None
        dic_balance = None
        dic_cf = None
        dic_income = None
        dic_fin_ind = None
        
        if self.freq == 1:
            # TODO : use fields = {field: kwargs} to enable params
            fields_market_daily = self._get_fields('market_daily', fields, append=True)
            if fields_market_daily:
                print "NOTE: price adjust method is [{:s} adjust]".format(self.adjust_mode)
                # no adjust prices and other market daily fields
                df_daily, msg1 = self.data_api.daily(symbol_str, start_date=self.extended_start_date_d, end_date=self.end_date,
                                                     adjust_mode=None, fields=sep.join(fields_market_daily))
                if self.all_price:
                    adj_cols = ['open', 'high', 'low', 'close', 'vwap']
                    # adjusted prices
                    df_daily_adjust, msg11 = self.data_api.daily(symbol_str, start_date=self.extended_start_date_d, end_date=self.end_date,
                                                                 adjust_mode=self.adjust_mode, fields=','.join(adj_cols))
                    df_daily_adjust = df_daily_adjust.loc[:, adj_cols]
                    # concat axis = 1
                    df_daily = df_daily.join(df_daily_adjust, rsuffix='_adj')
                    if msg11 != '0,':
                        print msg11
                if msg1 != '0,':
                    print msg1
                dic_market_daily = self._group_df_to_dict(df_daily, 'symbol')

            fields_ref_daily = self._get_fields('ref_daily', fields)
            if fields_ref_daily:
                df_ref_daily, msg2 = self.data_api.query_lb_dailyindicator(symbol_str, self.extended_start_date_d, self.end_date,
                                                                           sep.join(fields_ref_daily))
                if msg2 != '0,':
                    print msg2
                dic_ref_daily = self._group_df_to_dict(df_ref_daily, 'symbol')

            fields_income = self._get_fields('income', fields, append=True)
            if fields_income:
                df_income, msg3 = self.data_api.query_lb_fin_stat('income', symbol_str, self.extended_start_date_q, self.end_date,
                                                                  sep.join(fields_income), drop_dup_cols=['symbol', self.REPORT_DATE_FIELD_NAME])
                if msg3 != '0,':
                    print msg3
                dic_income = self._group_df_to_dict(df_income, 'symbol')

            fields_balance = self._get_fields('balance_sheet', fields, append=True)
            if fields_balance:
                df_balance, msg3 = self.data_api.query_lb_fin_stat('balance_sheet', symbol_str, self.extended_start_date_q, self.end_date,
                                                                   sep.join(fields_balance), drop_dup_cols=['symbol', self.REPORT_DATE_FIELD_NAME])
                if msg3 != '0,':
                    print msg3
                dic_balance = self._group_df_to_dict(df_balance, 'symbol')

            fields_cf = self._get_fields('cash_flow', fields, append=True)
            if fields_cf:
                df_cf, msg3 = self.data_api.query_lb_fin_stat('cash_flow', symbol_str, self.extended_start_date_q, self.end_date,
                                                              sep.join(fields_cf), drop_dup_cols=['symbol', self.REPORT_DATE_FIELD_NAME])
                if msg3 != '0,':
                    print msg3
                dic_cf = self._group_df_to_dict(df_cf, 'symbol')

            fields_fin_ind = self._get_fields('fin_indicator', fields, append=True)
            if fields_fin_ind:
                df_fin_ind, msg4 = self.data_api.query_lb_fin_stat('fin_indicator', symbol_str,
                                                                   self.extended_start_date_q, self.end_date,
                                                                   sep.join(fields_cf), drop_dup_cols=['symbol', self.REPORT_DATE_FIELD_NAME])
                if msg4 != '0,':
                    print msg4
                dic_fin_ind = self._group_df_to_dict(df_fin_ind, 'symbol')
        else:
            raise NotImplementedError("freq = {}".format(self.freq))
        
        return dic_market_daily, dic_ref_daily, dic_income, dic_balance, dic_cf, dic_fin_ind

    @staticmethod
    def _process_index(df, index_name='trade_date'):
        """
        Drop duplicates, set and sort index.
        
        Parameters
        ----------
        df : pd.DataFrame
            index of df must be unique.
        index_name : str, optional
            label of column which will be used as index.

        Returns
        -------
        df : pd.DataFrame
            processed
        
        Notes
        -----
        We do not use inplace operations, which will be a lot slower

        """
        # df.drop_duplicates(subset=index_name, inplace=True)  # TODO not a good solution
        dtype_idx = df.dtypes[index_name].type
        if not issubclass(dtype_idx, (int, np.integer)):
            df = df.astype(dtype={index_name: int})  # fast data type conversion
        
        dup = df.duplicated(subset=index_name)
        if np.sum(dup.values) > 0:
            # TODO
            print "WARNING: Duplicate {:s} encountered, droped.".format(index_name)
            idx = np.logical_not(dup)
            df = df.loc[idx, :]
        
        df = df.set_index(index_name)
        df = df.sort_index(axis=0)

        if 'symbol' in df.columns:
            df = df.drop(['symbol'], axis=1)
        
        return df

    def _dic_of_df_to_multi_index_df(self, dic, level_names=None):
        """
        Convert dict of DataFrame to MultiIndex DataFrame.
        Columns of result will be MultiIndex constructed using keys of dict and columns of DataFrame.
        Index of result will be the same with DataFrame.
        Different DataFrame will be aligned (outer join) using index.

        Parameters
        ----------
        dic : dict
            {symbol: DataFrame with index be datetime and columns be fields}.
        fields : list or np.ndarray
            Column labels for MultiIndex level 0.
        level_names : list of str
            Name of columns.

        Returns
        -------
        merge : pd.DataFrame
            with MultiIndex columns.

        """
        if level_names is None:
            level_names = ['symbol', 'field']
        merge = pd.concat(dic, axis=1)
        
        values = dic.values()
        idx = np.unique(np.concatenate([df.index.values for df in values]))
        fields = np.unique(np.concatenate([df.columns.values for df in values]))

        cols_multi = pd.MultiIndex.from_product([self.symbol, fields], names=level_names)
        cols_multi = cols_multi.sort_values()
        merge_final = pd.DataFrame(index=idx, columns=cols_multi, data=np.nan)

        # TODO: this step costs much time for big data
        merge_final.loc[merge.index, merge.columns] = merge  # index and column of merge, df must be the same

        if merge_final.isnull().sum().sum() > 0:
            print "WARNING: there are NaN values in your data, NO fill."
            # merge.fillna(method='ffill')

        if merge_final.shape != merge.shape:
            idx_diff = sorted(set(merge_final.index) - set(merge.index))
            col_diff = sorted(set(merge_final.columns.levels[0].values) - set(merge.columns.levels[0].values))
            print ("WARNING: some data is unavailable: "
                   + "\n    At index " + ', '.join(idx_diff)
                   + "\n    At fields " + ', '.join(col_diff))
        return merge_final

    def _preprocess_market_daily(self, dic):
        """
        Process data and construct MultiIndex.
        
        Parameters
        ----------
        dic : dict

        Returns
        -------
        res : pd.DataFrame

        """
        if not dic:
            return None
        for sec, df in dic.viewitems():
            # df = df.astype({'trade_status': str})
            dic[sec] = self._process_index(df, self.TRADE_DATE_FIELD_NAME)
            
        res = self._dic_of_df_to_multi_index_df(dic, level_names=['symbol', 'field'])
        return res
        
    def _preprocess_ref_daily(self, dic, fields):
        """
        Process data and construct MultiIndex.
        
        Parameters
        ----------
        dic : dict

        Returns
        -------
        res : pd.DataFrame

        """
        if not dic:
            return None
        for sec, df in dic.viewitems():
            df_mod = self._process_index(df, self.TRADE_DATE_FIELD_NAME)
            df_mod = df_mod.loc[:, self._get_fields('ref_daily', fields)]
            dic[sec] = df_mod
        
        res = self._dic_of_df_to_multi_index_df(dic, level_names=['symbol', 'field'])
        return res

    def _preprocess_ref_quarterly(self, type_, dic, fields):
        """
        Process data and construct MultiIndex.
        
        Parameters
        ----------
        dic : dict

        Returns
        -------
        res : pd.DataFrame

        """
        if not dic:
            return None
        new_dic = dict()
        for sec, df in dic.viewitems():
            df_mod = df.loc[:, self._get_fields(type_, fields, append=True)]
            df_mod = self._process_index(df_mod, self.REPORT_DATE_FIELD_NAME)
            
            new_dic[sec] = df_mod
    
        res = self._dic_of_df_to_multi_index_df(new_dic, level_names=['symbol', 'field'])
        return res
    
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
        dfs = [df for df in dfs if df is not None]
        if not dfs:
            return None
        
        merge = pd.concat(dfs, axis=1, join='outer')
        
        # drop duplicated columns. ONE LINE EFFICIENT version
        merge = merge.loc[:, ~merge.columns.duplicated()]
        
        if merge.isnull().sum().sum() > 0:
            print "WARNING: nan in final merged data. NO fill"
            # merge.fillna(method='ffill', inplace=True)
            
        merge = merge.sort_index(axis=1, level=['symbol', 'field'])
        merge.index.name = index_name
        
        return merge

    def _merge_data2(self, dfs):
        """
        Merge data from different APIs into one DataFrame.
        
        Parameters
        ----------
        trade_date : bool

        Returns
        -------
        merge : pd.DataFrame
        
        """
        # align on date index, concatenate on columns (symbol and fields)
        dfs = [df for df in dfs if df is not None]
        
        fields = []
        for df in dfs:
            fields.extend(df.columns.get_level_values(level='field').unique())
        col_multi = pd.MultiIndex.from_product([self.symbol, fields], names=['symbol', 'field'])
        merge = pd.DataFrame(index=dfs[0].index, columns=col_multi, data=None)
        
        for df in dfs:
            fields_df = df.columns.get_level_values(level='field')
            sec_df = df.columns.get_level_values(level='symbol')
            idx = df.index
            merge.loc[idx, pd.IndexSlice[sec_df, fields_df]] = df
            
        if merge.isnull().sum().sum() > 0:
            print "WARNING: nan in final merged data. NO fill"
            # merge.fillna(method='ffill', inplace=True)
    
        merge.sort_index(axis=1, level=['symbol', 'field'], inplace=True)
    
        return merge

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
    
    def _prepare_data(self, fields):
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
        print "Query data - query..."
        dic_market_daily, dic_ref_daily, dic_income, dic_balance_sheet, dic_cash_flow, dic_fin_ind = \
            self._query_data(self.symbol, fields)
        
        # pre-process data
        print "Query data - preprocess..."
        multi_market_daily = self._preprocess_market_daily(dic_market_daily)
        multi_ref_daily = self._preprocess_ref_daily(dic_ref_daily, fields)
        multi_income = self._preprocess_ref_quarterly('income', dic_income, fields)
        multi_balance_sheet = self._preprocess_ref_quarterly('balance_sheet', dic_balance_sheet, fields)
        multi_cash_flow = self._preprocess_ref_quarterly('cash_flow', dic_cash_flow, fields)
        multi_fin_ind = self._preprocess_ref_quarterly('fin_indicator', dic_fin_ind, fields)
    
        print "Query data - merge..."
        merge_d = self._merge_data([multi_market_daily, multi_ref_daily],
                                   index_name=self.TRADE_DATE_FIELD_NAME)
        merge_q = self._merge_data([multi_income, multi_balance_sheet, multi_cash_flow, multi_fin_ind],
                                   index_name=self.REPORT_DATE_FIELD_NAME)
    
        # drop dates that are not trade date
        if merge_d is not None:
            trade_dates = self.dates
            merge_d = merge_d.loc[trade_dates, pd.IndexSlice[:, :]].copy()
        
        return merge_d, merge_q
    
    def _prepare_adj_factor(self):
        """Query and append daily adjust factor for prices."""
        # TODO if all symbols are index, we do not fetch adj_factor
        symbol_str = ','.join(self.symbol)
        df_adj = self.data_api.get_adj_factor_daily(symbol_str,
                                                    start_date=self.extended_start_date_d, end_date=self.end_date, div=False)
        self.append_df(df_adj, 'adjust_factor', is_quarterly=False)

    def _prepare_comp_info(self):
        df = self.data_api.get_index_comp_df(self.universe, self.extended_start_date_d, self.end_date)
        self.append_df(df, 'index_member', is_quarterly=False)
    
    def _prepare_inst_info(self):
        res = self.data_api.query_inst_info(symbol=','.join(self.symbol),
                                            fields='symbol,inst_type,name,list_date,'
                                                   'delist_date,product,pricetick,buylot,setlot',
                                            inst_type="")
        self._data_inst = res

    def prepare_data(self):
        """Prepare data for the FIRST time."""
        # prepare benchmark and group
        print "Query data..."
        self.data_d, self.data_q = self._prepare_data(self.fields)

        print "Query instrument info..."
        self._prepare_inst_info()
        
        print "Query adj_factor..."
        self._prepare_adj_factor()

        if self.universe:
            print "Query benchmark..."
            self._data_benchmark = self._prepare_benchmark()
            print "Query benchmar member info..."
            self._prepare_comp_info()
            
            group_fields = self._get_fields('group', self.fields)
            if group_fields:
                print "Query groups (industry)..."
                self._prepare_group(group_fields)

        print "Data has been successfully prepared."

    def init_from_config(self, props, data_api):
        """
        Query various data from data_server and automatically merge them.
        This make research / trade easier.
        
        Parameters
        ----------
        props : dict, optional
            start_date, end_date, freq, symbol, fields
        data_api : BaseDataServer
        
        """
        self.data_api = data_api
    
        sep = ','
    
        # initialize parameters
        self.start_date = props['start_date']
        self.extended_start_date_d = dtutil.shift(self.start_date, n_weeks=-8)  # query more data
        self.extended_start_date_q = dtutil.shift(self.start_date, n_weeks=-80)
        self.end_date = props['end_date']
        self.all_price = props.get('all_price', True)
        
        # get and filter fields
        fields = props.get('fields', [])
        if fields:
            fields = props['fields'].split(sep)
            self.fields = [field for field in fields if self._is_predefined_field(field)]
            if len(self.fields) < len(fields):
                print "Field name [{}] not valid, ignore.".format(set.difference(set(fields), set(self.fields)))
        
        # TODO: hard-coded
        if self.all_price:
            self.fields.extend(['open_adj', 'high_adj', 'low_adj', 'close_adj'])
        
        self.freq = props['freq']
        self.universe = props.get('universe', "")
        if self.universe:
            self.symbol = data_api.get_index_comp(self.universe, self.extended_start_date_d, self.end_date)
            self.fields.append('index_member')
        else:
            self.symbol = props['symbol'].split(sep)  # list
        self.symbol = sorted(self.symbol)
    
        print "Initialize config success."
        
    def _prepare_benchmark(self):
        df_bench, msg = self.data_api.daily(self.universe,
                                            start_date=self.extended_start_date_d, end_date=self.end_date,
                                            adjust_mode=self.adjust_mode, fields='close')
        if msg != '0,':
            raise ValueError("msg = {:s}".format(msg))
        
        df_bench = self._process_index(df_bench, self.TRADE_DATE_FIELD_NAME)
        return df_bench
    
    def _prepare_group(self, group_fields):
        data_map = {'sw1': ('SW', 1),
                    'sw2': ('SW', 2),
                    'sw3': ('SW', 3),
                    'sw4': ('SW', 4),
                    'zz1': ('ZZ', 1),
                    'zz2': ('ZZ', 2)}
        for field in group_fields:
            type_, level = data_map[field]
            df = self.data_api.get_industry_daily(symbol=','.join(self.symbol),
                                                  start_date=self.extended_start_date_q, end_date=self.end_date,
                                                  type_=type_, level=level)
            self.append_df(df, field, is_quarterly=False)
    
    def _add_field(self, field_name, is_quarterly=None):
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
                print "Add field failed. No data_api available. Please specify one in parameter."
                return False
        else:
            self.data_api = data_api
            
        if field_name in self.fields:
            print "Field name [{:s}] already exists.".format(field_name)
            return False

        if not self._is_predefined_field(field_name):
            print "Field name [{}] not valid, ignore.".format(field_name)
            return False

        merge_d, merge_q = self._prepare_data([field_name])
    
        # auto decide whether is quarterly
        is_quarterly = merge_q is not None
        if is_quarterly:
            merge = merge_q
        else:
            merge = merge_d
            
        merge = merge.loc[:, pd.IndexSlice[:, field_name]]
        self.append_df(merge, field_name, is_quarterly=is_quarterly)  # whether contain only trade days is decided by existing data.
        
        return True
    
    def add_formula(self, field_name, formula, is_quarterly, formula_func_name_style='upper', data_api=None):
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
        formula_func_name_style : {'upper', 'lower'}, optional
        data_api : RemoteDataService, optional
        
        """
        if data_api is not None:
            self.data_api = data_api
            
        if field_name in self.fields:
            print "Add formula failed: name [{:s}] exist. Try another name.".format(field_name)
            return
        
        parser = Parser()
        parser.set_capital(formula_func_name_style)
        
        expr = parser.parse(formula)
        
        var_df_dic = dict()
        var_list = expr.variables()
        
        # TODO
        # users do not need to prepare data before add_formula
        if not self.fields:
            self.fields.extend(var_list)
            self.prepare_data()
        else:
            for var in var_list:
                if var not in self.fields:
                    print "Variable [{:s}] is not recognized (it may be wrong)," \
                          "try to fetch from the server...".format(var)
                    success = self.add_field(var)
                    if not success:
                        return
        
        df_ann = self.get_ann_df()
        for var in var_list:
            if self._is_quarter_field(var):
                df_var = self.get_ts_quarter(var, start_date=self.extended_start_date_q)
            else:
                # must use extended date. Default is start_date
                df_var = self.get_ts(var, start_date=self.extended_start_date_d, end_date=self.end_date)
            
            var_df_dic[var] = df_var
        
        # TODO: send ann_date into expr.evaluate. We assume that ann_date of all fields of a symbol is the same
        df_eval = parser.evaluate(var_df_dic, ann_dts=df_ann, trade_dts=self.dates)
        
        self.append_df(df_eval, field_name, is_quarterly=is_quarterly)

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
        
    def load_dataview(self, folder='.'):
        """
        Load data from local file.
        
        Parameters
        ----------
        folder : str, optional
            Folder path to store hd5 file and meta data.
            
        """
        meta_data = jaqs.util.fileio.read_json(os.path.join(folder, 'meta_data.json'))
        dic = self._load_h5(os.path.join(folder, 'data.hd5'))
        self.data_d = dic.get('/data_d', None)
        self.data_q = dic.get('/data_q', None)
        self._data_benchmark = dic.get('/data_benchmark', None)
        self._data_inst = dic.get('/data_inst', None)
        self.__dict__.update(meta_data)
        
        print "Dataview loaded successfully."

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
            res = self.data_api.get_trade_date(self.extended_start_date_d, self.end_date, is_datetime=False)
        else:
            raise ValueError("Cannot get dates array when neither of data and data_api exists.")
            
        return res

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
            fields = self.fields
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
        
        fields_daily = self._get_fields('daily', fields)
        fields_quarterly = self._get_fields('quarterly', fields)
        if not fields_daily and not fields_quarterly:
            return None
        
        # get df_daily and df_quarterly from data_d and data_q
        df_quarterly_expanded = None
        if fields_quarterly:
            df_ref_quarterly = self.data_q.loc[:,
                                               pd.IndexSlice[symbol, fields_quarterly]]
            df_ref_ann = self.data_q.loc[:,
                                         pd.IndexSlice[symbol, self.ANN_DATE_FIELD_NAME]]
            df_ref_ann.columns = df_ref_ann.columns.droplevel(level='field')
            
            dic_expanded = dict()
            for field_name, df in df_ref_quarterly.groupby(level=1, axis=1):  # by column multiindex fields
                df_expanded = align(df, df_ref_ann, self.dates)
                dic_expanded[field_name] = df_expanded
            df_quarterly_expanded = pd.concat(dic_expanded.values(), axis=1)
            df_quarterly_expanded.index.name = self.TRADE_DATE_FIELD_NAME
            df_quarterly_expanded = df_quarterly_expanded.loc[start_date: end_date, :]
        
        df_daily = None
        if fields_daily:
            df_daily = self.data_d.loc[pd.IndexSlice[start_date: end_date],
                                       pd.IndexSlice[symbol, fields_daily]]
        
        # res will be one of daily/quarterly, or combination of them.
        if fields_daily and fields_quarterly:
            res = self._merge_data([df_daily, df_quarterly_expanded], index_name=self.TRADE_DATE_FIELD_NAME)
        else:
            res = df_quarterly_expanded if df_quarterly_expanded is not None else df_daily
        
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
            print "No data. for date={}, fields={}, symbol={}".format(snapshot_date, fields, symbol)
            return
        
        res = res.stack(level='symbol', dropna=False)
        res.index = res.index.droplevel(level=self.TRADE_DATE_FIELD_NAME)
        
        return res

    def get_ann_df(self):
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
        
        df_ann = df_ann.copy()
        df_ann.columns = df_ann.columns.droplevel(level='field')
    
        return df_ann
        
    def get_ts_quarter(self, field, symbol="", start_date=0, end_date=0):
        # TODO
        sep = ','
        if not symbol:
            symbol = self.symbol
        else:
            symbol = symbol.split(sep)
    
        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date
    
        df_ref_quarterly = self.data_q.loc[:,
                                           pd.IndexSlice[symbol, field]]
        df_ref_quarterly.columns = df_ref_quarterly.columns.droplevel(level='field')
        
        return df_ref_quarterly
    
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
            print "No data. for start_date={}, end_date={}, field={}, symbol={}".format(start_date,
                                                                                         end_date, field, symbol)
            raise ValueError
            return
        
        res.columns = res.columns.droplevel(level='field')
        
        return res

    def save_dataview(self, folder_path=".", sub_folder=""):
        """
        Save data and meta_data_to_store to a single hd5 file.
        Store at output/sub_folder
        
        Parameters
        ----------
        folder_path : str
        sub_folder : str

        """
        if not sub_folder:
            sub_folder = "{:d}_{:d}_freq={:d}D".format(self.start_date, self.end_date, self.freq)
        
        folder_path = os.path.join(folder_path, sub_folder)
        abs_folder = os.path.abspath(folder_path)
        meta_path = os.path.join(folder_path, 'meta_data.json')
        data_path = os.path.join(folder_path, 'data.hd5')
        
        data_to_store = {'data_d': self.data_d, 'data_q': self.data_q,
                         'data_benchmark': self.data_benchmark, 'data_inst': self.data_inst}
        data_to_store = {k: v for k, v in data_to_store.viewitems() if v is not None}
        meta_data_to_store = {key: self.__dict__[key] for key in self.meta_data_list}

        print "\nStore data..."
        jaqs.util.fileio.save_json(meta_data_to_store, meta_path)
        self._save_h5(data_path, data_to_store)
        
        print ("Dataview has been successfully saved to:\n"
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
        
        jaqs.util.fileio.create_dir(fp)
        h5 = pd.HDFStore(fp)
        for key, value in dic.viewitems():
            h5[key] = value
        h5.close()
    
    def append_df(self, df, field_name, is_quarterly=False):
        """
        Append DataFrame to existing multi-index DataFrame and add corresponding field name.
        
        Parameters
        ----------
        df : pd.DataFrame or pd.Series
        field_name : str
        is_quarterly : bool
            Whether df is quarterly data (like quarterly financial statement) or daily data.

        """
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
            
        multi_idx = pd.MultiIndex.from_product([the_data.columns.levels[0], [field_name]])
        df.columns = multi_idx
        
        merge = the_data.join(df, how='left')  # left: keep index of existing data unchanged
        merge.sort_index(axis=1, level=['symbol', 'field'], inplace=True)

        if is_quarterly:
            self.data_q = merge
        else:
            self.data_d = merge
        self._add_field(field_name, is_quarterly)
    
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
