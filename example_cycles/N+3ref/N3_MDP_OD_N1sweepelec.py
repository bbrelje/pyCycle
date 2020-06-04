import numpy as np
import time
import pickle
from pprint import pprint

import openmdao.api as om

from N3ref import N3, viewer, dump_guess, load_guess

prob = om.Problem()

des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

des_vars.add_output('inlet:ram_recovery', 0.9980),
des_vars.add_output('fan:PRdes', 1.28507887),
des_vars.add_output('fan:effDes', 0.96888),
des_vars.add_output('fan:effPoly', 0.97),
des_vars.add_output('splitter:BPR', 23.7281), #23.9878
des_vars.add_output('duct2:dPqP', 0.0100),
des_vars.add_output('lpc:PRdes', 4.000),
des_vars.add_output('lpc:effDes', 0.889513),
des_vars.add_output('lpc:effPoly', 0.905),
des_vars.add_output('duct25:dPqP', 0.0150),
des_vars.add_output('hpc:PRdes', 14.103),
des_vars.add_output('OPR', 53.6332) #53.635)
des_vars.add_output('OPR_simple', 60.70942491)
des_vars.add_output('hpc:effDes', 0.847001),
des_vars.add_output('hpc:effPoly', 0.89),
des_vars.add_output('burner:dPqP', 0.0400),
des_vars.add_output('hpt:effDes', 0.922649),
des_vars.add_output('hpt:effPoly', 0.91),
des_vars.add_output('duct45:dPqP', 0.0050),
des_vars.add_output('lpt:effDes', 0.940104),
des_vars.add_output('lpt:effPoly', 0.92),
des_vars.add_output('duct5:dPqP', 0.0100),
des_vars.add_output('core_nozz:Cv', 0.9999),
des_vars.add_output('duct17:dPqP', 0.0150),
des_vars.add_output('byp_nozz:Cv', 0.9975),
des_vars.add_output('fan_shaft:Nmech', 2184.5, units='rpm'),
des_vars.add_output('lp_shaft:Nmech', 6772.0, units='rpm'),
des_vars.add_output('lp_shaft:fracLoss', 0.01)
des_vars.add_output('hp_shaft:Nmech', 20871.0, units='rpm'),
des_vars.add_output('hp_shaft:HPX', 350.0, units='hp'),

des_vars.add_output('bld25:sbv:frac_W', 0.0),
des_vars.add_output('hpc:bld_inlet:frac_W', 0.0),
des_vars.add_output('hpc:bld_inlet:frac_P', 0.1465),
des_vars.add_output('hpc:bld_inlet:frac_work', 0.5),
des_vars.add_output('hpc:bld_exit:frac_W', 0.02),
des_vars.add_output('hpc:bld_exit:frac_P', 0.1465),
des_vars.add_output('hpc:bld_exit:frac_work', 0.5),
des_vars.add_output('hpc:cust:frac_W', 0.0),
des_vars.add_output('hpc:cust:frac_P', 0.1465),
des_vars.add_output('hpc:cust:frac_work', 0.35),
des_vars.add_output('bld3:bld_inlet:frac_W', 0.05943196), #different than NPSS due to Wref
des_vars.add_output('bld3:bld_exit:frac_W', 0.05742255), #different than NPSS due to Wref
des_vars.add_output('hpt:bld_inlet:frac_P', 1.0),
des_vars.add_output('hpt:bld_exit:frac_P', 0.0),
des_vars.add_output('lpt:bld_inlet:frac_P', 1.0),
des_vars.add_output('lpt:bld_exit:frac_P', 0.0),
des_vars.add_output('bypBld:frac_W', 0.0),

des_vars.add_output('inlet:MN_out', 0.625),
des_vars.add_output('fan:MN_out', 0.45)
des_vars.add_output('splitter:MN_out1', 0.45)
des_vars.add_output('splitter:MN_out2', 0.45)
des_vars.add_output('duct2:MN_out', 0.45),
des_vars.add_output('lpc:MN_out', 0.45),
des_vars.add_output('bld25:MN_out', 0.45),
des_vars.add_output('duct25:MN_out', 0.45),
des_vars.add_output('hpc:MN_out', 0.30),
des_vars.add_output('bld3:MN_out', 0.30)
des_vars.add_output('burner:MN_out', 0.10),
des_vars.add_output('hpt:MN_out', 0.30),
des_vars.add_output('duct45:MN_out', 0.45),
des_vars.add_output('lpt:MN_out', 0.35),
des_vars.add_output('duct5:MN_out', 0.25),
des_vars.add_output('bypBld:MN_out', 0.45),
des_vars.add_output('duct17:MN_out', 0.45),

# POINT 1: Top-of-climb (TOC)
des_vars.add_output('TOC:alt', 35000., units='ft'),
des_vars.add_output('TOC:MN', 0.8),
des_vars.add_output('TOC:T4max', 3150.0, units='degR'),
des_vars.add_output('TOC:Fn_des', 6073.4, units='lbf'),
des_vars.add_output('TOC:ram_recovery', 0.9980),
des_vars.add_output('TOC:BPR', 20.96364424, units=None)
des_vars.add_output('TOC:W', 810.91772237, units='lbm/s')
des_vars.add_output('TR', 0.91024208)

# POINT 2: Rolling Takeoff (RTO)
des_vars.add_output('RTO:MN', 0.25),
des_vars.add_output('RTO:alt', 0.0, units='ft'),
des_vars.add_output('RTO:Fn_target', 22800.0, units='lbf'), #8950.0
des_vars.add_output('RTO:dTs', 27.0, units='degR')
des_vars.add_output('RTO:Ath', 5532.3, units='inch**2')
des_vars.add_output('RTO:RlineMap', 1.75)
des_vars.add_output('RTO:T4max', 3233.16173465, units='degR')
des_vars.add_output('RTO:W', 1916.13, units='lbm/s')
des_vars.add_output('RTO:ram_recovery', 0.9970),
des_vars.add_output('RTO:duct2:dPqP', 0.0073)
des_vars.add_output('RTO:duct25:dPqP', 0.0138)
des_vars.add_output('RTO:duct45:dPqP', 0.0051)
des_vars.add_output('RTO:duct5:dPqP', 0.0058)
des_vars.add_output('RTO:duct17:dPqP', 0.0132)

# POINT 3: Sea-Level Static (SLS)
des_vars.add_output('SLS:MN', 0.001),
des_vars.add_output('SLS:alt', 0.0, units='ft'),
des_vars.add_output('SLS:Fn_target', 28620.9, units='lbf'), #8950.0
des_vars.add_output('SLS:dTs', 27.0, units='degR')
des_vars.add_output('SLS:Ath', 6315.6, units='inch**2')
des_vars.add_output('SLS:RlineMap', 1.75)
des_vars.add_output('SLS:ram_recovery', 0.9950),
des_vars.add_output('SLS:duct2:dPqP', 0.0058)
des_vars.add_output('SLS:duct25:dPqP', 0.0126)
des_vars.add_output('SLS:duct45:dPqP', 0.0052)
des_vars.add_output('SLS:duct5:dPqP', 0.0043)
des_vars.add_output('SLS:duct17:dPqP', 0.0123)

# POINT 4: Cruise (CRZ)
des_vars.add_output('CRZ:MN', 0.8),
des_vars.add_output('CRZ:alt', 35000.0, units='ft'),
des_vars.add_output('CRZ:Fn_target', 5220., units='lbf'), #8950.0
des_vars.add_output('CRZ:dTs', 0.0, units='degR')
des_vars.add_output('CRZ:Ath', 4747.1, units='inch**2')
des_vars.add_output('CRZ:RlineMap', 1.9397)
des_vars.add_output('CRZ:ram_recovery', 0.9980),
des_vars.add_output('CRZ:duct2:dPqP', 0.0092)
des_vars.add_output('CRZ:duct25:dPqP', 0.0138)
des_vars.add_output('CRZ:duct45:dPqP', 0.0050)
des_vars.add_output('CRZ:duct5:dPqP', 0.0097)
des_vars.add_output('CRZ:duct17:dPqP', 0.0148)
des_vars.add_output('CRZ:VjetRatio', 1.35) #1.41038)
des_vars.add_output('CRZ:Wfuel', 0.6395, units='lbm/s')
des_vars.add_output('CRZ:N1max', 6800., units='rpm')

# TOC POINT (DESIGN)
prob.model.add_subsystem('TOC', N3())

prob.model.connect('TOC:alt', 'TOC.fc.alt')
prob.model.connect('TOC:MN', 'TOC.fc.MN')

prob.model.connect('TOC:ram_recovery', 'TOC.inlet.ram_recovery')
prob.model.connect('fan:PRdes', ['TOC.fan.PR', 'TOC.opr_calc.FPR'])
prob.model.connect('fan:effPoly', 'TOC.balance.rhs:fan_eff')
prob.model.connect('duct2:dPqP', 'TOC.duct2.dPqP')
prob.model.connect('lpc:PRdes', ['TOC.lpc.PR', 'TOC.opr_calc.LPCPR'])
prob.model.connect('lpc:effPoly', 'TOC.balance.rhs:lpc_eff')
prob.model.connect('duct25:dPqP', 'TOC.duct25.dPqP')
prob.model.connect('OPR_simple', 'TOC.balance.rhs:hpc_PR')
prob.model.connect('burner:dPqP', 'TOC.burner.dPqP')
prob.model.connect('hpt:effPoly', 'TOC.balance.rhs:hpt_eff')
prob.model.connect('duct45:dPqP', 'TOC.duct45.dPqP')
prob.model.connect('lpt:effPoly', 'TOC.balance.rhs:lpt_eff')
prob.model.connect('duct5:dPqP', 'TOC.duct5.dPqP')
prob.model.connect('core_nozz:Cv', ['TOC.core_nozz.Cv', 'TOC.ext_ratio.core_Cv'])
prob.model.connect('duct17:dPqP', 'TOC.duct17.dPqP')
prob.model.connect('byp_nozz:Cv', ['TOC.byp_nozz.Cv', 'TOC.ext_ratio.byp_Cv'])
prob.model.connect('fan_shaft:Nmech', 'TOC.Fan_Nmech')
prob.model.connect('lp_shaft:Nmech', 'TOC.LP_Nmech')
prob.model.connect('lp_shaft:fracLoss', 'TOC.lp_shaft.fracLoss')
prob.model.connect('hp_shaft:Nmech', 'TOC.HP_Nmech')
prob.model.connect('hp_shaft:HPX', 'TOC.hp_shaft.HPX')

prob.model.connect('bld25:sbv:frac_W', 'TOC.bld25.sbv:frac_W')
prob.model.connect('hpc:bld_inlet:frac_W', 'TOC.hpc.bld_inlet:frac_W')
prob.model.connect('hpc:bld_inlet:frac_P', 'TOC.hpc.bld_inlet:frac_P')
prob.model.connect('hpc:bld_inlet:frac_work', 'TOC.hpc.bld_inlet:frac_work')
prob.model.connect('hpc:bld_exit:frac_W', 'TOC.hpc.bld_exit:frac_W')
prob.model.connect('hpc:bld_exit:frac_P', 'TOC.hpc.bld_exit:frac_P')
prob.model.connect('hpc:bld_exit:frac_work', 'TOC.hpc.bld_exit:frac_work')
prob.model.connect('hpc:cust:frac_W', 'TOC.hpc.cust:frac_W')
prob.model.connect('hpc:cust:frac_P', 'TOC.hpc.cust:frac_P')
prob.model.connect('hpc:cust:frac_work', 'TOC.hpc.cust:frac_work')
prob.model.connect('hpt:bld_inlet:frac_P', 'TOC.hpt.bld_inlet:frac_P')
prob.model.connect('hpt:bld_exit:frac_P', 'TOC.hpt.bld_exit:frac_P')
prob.model.connect('lpt:bld_inlet:frac_P', 'TOC.lpt.bld_inlet:frac_P')
prob.model.connect('lpt:bld_exit:frac_P', 'TOC.lpt.bld_exit:frac_P')
prob.model.connect('bypBld:frac_W', 'TOC.byp_bld.bypBld:frac_W')

prob.model.connect('inlet:MN_out', 'TOC.inlet.MN')
prob.model.connect('fan:MN_out', 'TOC.fan.MN')
prob.model.connect('splitter:MN_out1', 'TOC.splitter.MN1')
prob.model.connect('splitter:MN_out2', 'TOC.splitter.MN2')
prob.model.connect('duct2:MN_out', 'TOC.duct2.MN')
prob.model.connect('lpc:MN_out', 'TOC.lpc.MN')
prob.model.connect('bld25:MN_out', 'TOC.bld25.MN')
prob.model.connect('duct25:MN_out', 'TOC.duct25.MN')
prob.model.connect('hpc:MN_out', 'TOC.hpc.MN')
prob.model.connect('bld3:MN_out', 'TOC.bld3.MN')
prob.model.connect('burner:MN_out', 'TOC.burner.MN')
prob.model.connect('hpt:MN_out', 'TOC.hpt.MN')
prob.model.connect('duct45:MN_out', 'TOC.duct45.MN')
prob.model.connect('lpt:MN_out', 'TOC.lpt.MN')
prob.model.connect('duct5:MN_out', 'TOC.duct5.MN')
prob.model.connect('bypBld:MN_out', 'TOC.byp_bld.MN')
prob.model.connect('duct17:MN_out', 'TOC.duct17.MN')


# OTHER POINTS (OFF-DESIGN)
pts = ['CRZ']

# prob.model.connect('RTO:Fn_target', 'RTO.balance.rhs:FAR')

# prob.model.add_subsystem('RTO', N3(design=False, cooling=True))
# prob.model.add_subsystem('SLS', N3(design=False))
prob.model.add_subsystem('CRZ', N3(design=False))

for pt in pts:

    prob.model.connect(pt+':alt', pt+'.fc.alt')
    prob.model.connect(pt+':MN', pt+'.fc.MN')
    prob.model.connect(pt+':dTs', pt+'.fc.dTs')
    prob.model.connect(pt+':RlineMap',pt+'.balance.rhs:BPR')

    prob.model.connect(pt+':ram_recovery', pt+'.inlet.ram_recovery')
    prob.model.connect('TOC.duct2.s_dPqP', pt+'.duct2.s_dPqP')
    prob.model.connect('TOC.duct25.s_dPqP', pt+'.duct25.s_dPqP')
    prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
    prob.model.connect('TOC.duct45.s_dPqP', pt+'.duct45.s_dPqP')
    prob.model.connect('TOC.duct5.s_dPqP', pt+'.duct5.s_dPqP')
    prob.model.connect('core_nozz:Cv', [pt+'.core_nozz.Cv', pt+'.ext_ratio.core_Cv'])
    prob.model.connect('TOC.duct17.s_dPqP', pt+'.duct17.s_dPqP')
    prob.model.connect('byp_nozz:Cv', [pt+'.byp_nozz.Cv', pt+'.ext_ratio.byp_Cv'])
    prob.model.connect('lp_shaft:fracLoss', pt+'.lp_shaft.fracLoss')
    prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')

    prob.model.connect('bld25:sbv:frac_W', pt+'.bld25.sbv:frac_W')
    prob.model.connect('hpc:bld_inlet:frac_W', pt+'.hpc.bld_inlet:frac_W')
    prob.model.connect('hpc:bld_inlet:frac_P', pt+'.hpc.bld_inlet:frac_P')
    prob.model.connect('hpc:bld_inlet:frac_work', pt+'.hpc.bld_inlet:frac_work')
    prob.model.connect('hpc:bld_exit:frac_W', pt+'.hpc.bld_exit:frac_W')
    prob.model.connect('hpc:bld_exit:frac_P', pt+'.hpc.bld_exit:frac_P')
    prob.model.connect('hpc:bld_exit:frac_work', pt+'.hpc.bld_exit:frac_work')
    prob.model.connect('hpc:cust:frac_W', pt+'.hpc.cust:frac_W')
    prob.model.connect('hpc:cust:frac_P', pt+'.hpc.cust:frac_P')
    prob.model.connect('hpc:cust:frac_work', pt+'.hpc.cust:frac_work')
    prob.model.connect('hpt:bld_inlet:frac_P', pt+'.hpt.bld_inlet:frac_P')
    prob.model.connect('hpt:bld_exit:frac_P', pt+'.hpt.bld_exit:frac_P')
    prob.model.connect('lpt:bld_inlet:frac_P', pt+'.lpt.bld_inlet:frac_P')
    prob.model.connect('lpt:bld_exit:frac_P', pt+'.lpt.bld_exit:frac_P')
    prob.model.connect('bypBld:frac_W', pt+'.byp_bld.bypBld:frac_W')

    prob.model.connect('TOC.fan.s_PR', pt+'.fan.s_PR')
    prob.model.connect('TOC.fan.s_Wc', pt+'.fan.s_Wc')
    prob.model.connect('TOC.fan.s_eff', pt+'.fan.s_eff')
    prob.model.connect('TOC.fan.s_Nc', pt+'.fan.s_Nc')
    prob.model.connect('TOC.lpc.s_PR', pt+'.lpc.s_PR')
    prob.model.connect('TOC.lpc.s_Wc', pt+'.lpc.s_Wc')
    prob.model.connect('TOC.lpc.s_eff', pt+'.lpc.s_eff')
    prob.model.connect('TOC.lpc.s_Nc', pt+'.lpc.s_Nc')
    prob.model.connect('TOC.hpc.s_PR', pt+'.hpc.s_PR')
    prob.model.connect('TOC.hpc.s_Wc', pt+'.hpc.s_Wc')
    prob.model.connect('TOC.hpc.s_eff', pt+'.hpc.s_eff')
    prob.model.connect('TOC.hpc.s_Nc', pt+'.hpc.s_Nc')
    prob.model.connect('TOC.hpt.s_PR', pt+'.hpt.s_PR')
    prob.model.connect('TOC.hpt.s_Wp', pt+'.hpt.s_Wp')
    prob.model.connect('TOC.hpt.s_eff', pt+'.hpt.s_eff')
    prob.model.connect('TOC.hpt.s_Np', pt+'.hpt.s_Np')
    prob.model.connect('TOC.lpt.s_PR', pt+'.lpt.s_PR')
    prob.model.connect('TOC.lpt.s_Wp', pt+'.lpt.s_Wp')
    prob.model.connect('TOC.lpt.s_eff', pt+'.lpt.s_eff')
    prob.model.connect('TOC.lpt.s_Np', pt+'.lpt.s_Np')

    prob.model.connect('TOC.gearbox.gear_ratio', pt+'.gearbox.gear_ratio')
    prob.model.connect('TOC.core_nozz.Throat:stat:area',pt+'.balance.rhs:W')

    prob.model.connect('TOC.inlet.Fl_O:stat:area', pt+'.inlet.area')
    prob.model.connect('TOC.fan.Fl_O:stat:area', pt+'.fan.area')
    prob.model.connect('TOC.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
    prob.model.connect('TOC.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
    prob.model.connect('TOC.duct2.Fl_O:stat:area', pt+'.duct2.area')
    prob.model.connect('TOC.lpc.Fl_O:stat:area', pt+'.lpc.area')
    prob.model.connect('TOC.bld25.Fl_O:stat:area', pt+'.bld25.area')
    prob.model.connect('TOC.duct25.Fl_O:stat:area', pt+'.duct25.area')
    prob.model.connect('TOC.hpc.Fl_O:stat:area', pt+'.hpc.area')
    prob.model.connect('TOC.bld3.Fl_O:stat:area', pt+'.bld3.area')
    prob.model.connect('TOC.burner.Fl_O:stat:area', pt+'.burner.area')
    prob.model.connect('TOC.hpt.Fl_O:stat:area', pt+'.hpt.area')
    prob.model.connect('TOC.duct45.Fl_O:stat:area', pt+'.duct45.area')
    prob.model.connect('TOC.lpt.Fl_O:stat:area', pt+'.lpt.area')
    prob.model.connect('TOC.duct5.Fl_O:stat:area', pt+'.duct5.area')
    prob.model.connect('TOC.byp_bld.Fl_O:stat:area', pt+'.byp_bld.area')
    prob.model.connect('TOC.duct17.Fl_O:stat:area', pt+'.duct17.area')


prob.model.connect('bld3:bld_exit:frac_W', 'TOC.bld3.bld_exit:frac_W')
prob.model.connect('bld3:bld_inlet:frac_W', 'TOC.bld3.bld_inlet:frac_W')

# prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'SLS.bld3.bld_exit:frac_W')
# prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'SLS.bld3.bld_inlet:frac_W')

prob.model.connect('bld3:bld_exit:frac_W', 'CRZ.bld3.bld_exit:frac_W')
prob.model.connect('bld3:bld_inlet:frac_W', 'CRZ.bld3.bld_inlet:frac_W')


# bal = prob.model.add_subsystem('bal', om.BalanceComp())
# bal.add_balance('TOC_BPR', val=23.7281, units=None, eq_units=None)
# prob.model.connect('bal.TOC_BPR', 'TOC.splitter.BPR')
# prob.model.connect('CRZ.ext_ratio.ER', 'bal.lhs:TOC_BPR')
# prob.model.connect('CRZ:VjetRatio', 'bal.rhs:TOC_BPR')
prob.model.connect('TOC:BPR', 'TOC.splitter.BPR')

# bal.add_balance('TOC_W', val=820.95, units='lbm/s', eq_units='degR')
# prob.model.connect('bal.TOC_W', 'TOC.fc.W')
# prob.model.connect('RTO.burner.Fl_O:tot:T', 'bal.lhs:TOC_W')
# prob.model.connect('RTO:T4max','bal.rhs:TOC_W')

prob.model.connect('TOC:W', 'TOC.fc.W')


# bal.add_balance('SLS_Fn_target', val=28620.8, units='lbf', eq_units='lbf', use_mult=True, mult_val=1.2553, ref0=28000.0, ref=30000.0)
# prob.model.connect('bal.SLS_Fn_target', 'SLS.balance.rhs:FAR')
# prob.model.connect('RTO.perf.Fn', 'bal.lhs:SLS_Fn_target')
# prob.model.connect('SLS.perf.Fn','bal.rhs:SLS_Fn_target')

prob.model.add_subsystem('T4_ratio',
                         om.ExecComp('TOC_T4 = RTO_T4*TR',
                                     RTO_T4={'value': 3400.0, 'units':'degR'},
                                     TOC_T4={'value': 3150.0, 'units':'degR'},
                                     TR={'value': 0.926470588, 'units': None}))
prob.model.connect('RTO:T4max','T4_ratio.RTO_T4')
prob.model.connect('T4_ratio.TOC_T4', 'TOC.balance.rhs:FAR')
prob.model.connect('TR', 'T4_ratio.TR')

# prob.model.add_subsystem('eng_op_limits',
#                          om.ExecComp('lim_ratio = max(T4/T4max, N1/N1max, N2/N2max)',
#                                      lim_ratio={'value': 0.95, 'units':None},
#                                      T4={'value': 2800., 'units':'degR'},
#                                      T4max={'value': 3233, 'units': 'degR'},
#                                      N1={'value': 6600., 'units': 'rpm'},
#                                      N1max={'value': 6900., 'units': 'rpm'},
#                                      N2={'value': 20000., 'units': 'rpm'},
#                                      N2max={'value': 22500., 'units': 'rpm'}))
# prob.model.connect('CRZ.burner.Fl_O:tot:T', 'eng_op_limits.T4')
# prob.model.connect('CRZ.balance.lp_Nmech', 'eng_op_limits.N1')
# prob.model.connect('CRZ.balance.hp_Nmech', 'eng_op_limits.N2')


# bal.add_balance('CRZ_Fn_target', val=5220., units='lbf', eq_units=None, rhs_val=1.0, ref0=5000.0, ref=7000.0)
prob.model.connect('CRZ:Fn_target', 'CRZ.balance.rhs:FAR')
# prob.model.connect('eng_op_limits.lim_ratio', 'bal.lhs:CRZ_Fn_target')

prob.model.set_order(['des_vars', 'T4_ratio', 'TOC', 'CRZ'])


newton = prob.model.nonlinear_solver = om.NewtonSolver()
newton.options['atol'] = 1e-6
newton.options['rtol'] = 1e-6
newton.options['iprint'] = 2
newton.options['maxiter'] = 20
newton.options['solve_subsystems'] = True
newton.options['max_sub_solves'] = 10
newton.options['err_on_non_converge'] = True
newton.options['reraise_child_analysiserror'] = False
newton.linesearch =  om.BoundsEnforceLS()
newton.linesearch.options['bound_enforcement'] = 'scalar'
newton.linesearch.options['iprint'] = -1

prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

# setup the optimization
prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
prob.driver.opt_settings={'Major step limit': 0.05}

prob.model.add_design_var('fan:PRdes', lower=1.20, upper=1.4)
prob.model.add_design_var('lpc:PRdes', lower=2.5, upper=4.0)
prob.model.add_design_var('OPR_simple', lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
prob.model.add_design_var('RTO:T4max', lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)
prob.model.add_design_var('CRZ:VjetRatio', lower=1.35, upper=1.45, ref0=1.35, ref=1.45)
prob.model.add_design_var('TR', lower=0.5, upper=0.95, ref0=0.5, ref=0.95)


prob.model.add_objective('CRZ.burner.Wfuel', ref0=.6, ref=0.7)
# to add the constraint to the model
prob.model.add_constraint('TOC.fan_dia.FanDia', upper=100.0, ref=100.0)
prob.model.add_constraint('TOC.perf.Fn', lower=5800.0, ref=6000.0)


# recorder = SqliteRecorder('N3_MDP_opt_model.sql')
# prob.model.add_recorder(recorder)
# prob.model.recording_options['record_inputs'] = True
# prob.model.recording_options['record_outputs'] = True
# prob.model.recording_options['record_model_metadata'] = False


# recorder2 = SqliteRecorder('N3_MDP_opt_driver.sql')
# prob.driver.add_recorder(recorder2)
# prob.driver.recording_options['includes'] = []
# prob.driver.recording_options['record_objectives'] = True
# prob.driver.recording_options['record_constraints'] = True
# prob.driver.recording_options['record_desvars'] = True
# prob.driver.recording_options['record_derivatives'] = True

prob.setup(check=False)

# prob['RTO.hpt_cooling.x_factor'] = 0.9

# initial guesses
prob['TOC.balance.FAR'] = 0.02650
# prob['bal.TOC_W'] = 820.95
prob['TOC.balance.lpt_PR'] = 10.937
prob['TOC.balance.hpt_PR'] = 4.185
prob['TOC.fc.balance.Pt'] = 5.272
prob['TOC.fc.balance.Tt'] = 444.41

DNAME = 'TOC'
ODNAMES = ['CRZ']
DPARMS = ['balance.FAR', 'balance.lpt_PR', 'balance.hpt_PR', 'fc.balance.Pt', 'fc.balance.Tt']
ODPARMS = ['balance.FAR',
           'balance.W',
           'balance.BPR',
           'balance.fan_Nmech',
           'balance.lp_Nmech',
           'balance.hp_Nmech',
           'fc.balance.Pt',
           'fc.balance.Tt',
           'hpt.PR',
           'lpt.PR',
           'fan.map.RlineMap',
           'lpc.map.RlineMap',
           'hpc.map.RlineMap',
           'gearbox.trq_base']

for pt in pts:

    if pt == 'RTO':
        prob[pt+'.balance.FAR'] = 0.02832
        prob[pt+'.balance.W'] = 1916.13
        prob[pt+'.balance.BPR'] = 25.5620
        prob[pt+'.balance.fan_Nmech'] = 2132.6
        prob[pt+'.balance.lp_Nmech'] = 6611.2
        prob[pt+'.balance.hp_Nmech'] = 22288.2
        prob[pt+'.fc.balance.Pt'] = 15.349
        prob[pt+'.fc.balance.Tt'] = 552.49
        prob[pt+'.hpt.PR'] = 4.210
        prob[pt+'.lpt.PR'] = 8.161
        prob[pt+'.fan.map.RlineMap'] = 1.7500
        prob[pt+'.lpc.map.RlineMap'] = 2.0052
        prob[pt+'.hpc.map.RlineMap'] = 2.0589
        prob[pt+'.gearbox.trq_base'] = 52509.1

    if pt == 'SLS':
        prob[pt+'.balance.FAR'] = 0.02541
        prob[pt+'.balance.W'] = 2000 # 1734.44
        prob[pt+'.balance.BPR'] = 27.3467
        prob[pt+'.balance.fan_Nmech'] = 1953.1
        prob[pt+'.balance.lp_Nmech'] = 6054.5
        prob[pt+'.balance.hp_Nmech'] = 21594.0
        prob[pt+'.fc.balance.Pt'] = 14.696
        prob[pt+'.fc.balance.Tt'] = 545.67
        prob[pt+'.hpt.PR'] = 4.245
        prob[pt+'.lpt.PR'] = 7.001
        prob[pt+'.fan.map.RlineMap'] = 1.7500
        prob[pt+'.lpc.map.RlineMap'] = 1.8632
        prob[pt+'.hpc.map.RlineMap'] = 2.0281
        prob[pt+'.gearbox.trq_base'] = 41779.4

    if pt == 'CRZ':
        prob[pt+'.balance.FAR'] = 0.02510
        prob[pt+'.balance.W'] = 802.79
        prob[pt+'.balance.BPR'] = 24.3233
        prob[pt+'.balance.fan_Nmech'] = 2118.7
        prob[pt+'.balance.lp_Nmech'] = 6567.9
        prob[pt+'.balance.hp_Nmech'] = 20574.1
        prob[pt+'.fc.balance.Pt'] = 5.272
        prob[pt+'.fc.balance.Tt'] = 444.41
        prob[pt+'.hpt.PR'] = 4.197
        prob[pt+'.lpt.PR'] = 10.803
        prob[pt+'.fan.map.RlineMap'] = 1.9397
        prob[pt+'.lpc.map.RlineMap'] = 2.1075
        prob[pt+'.hpc.map.RlineMap'] = 1.9746
        prob[pt+'.gearbox.trq_base'] = 22369.7


st = time.time()

prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)
load_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'initpklod.pkl')
# prob.run_model()
# dump_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'initpklod.pkl')


# for alt in [35000., 32500., 30000., 27500., 25000., 22500.]:
#     for i, thrust in enumerate([5220., 4800., 4400., 4200., 3800., 3400., 3000.]):

#         print('======Running Thrust ' +str(thrust)+' Alt '+str(alt)+'===================')
#         prob['CRZ:Fn_target'] =  thrust
#         prob['CRZ:alt'] = alt
#         prob.run_model()
#         for pt in ['TOC']+pts:
#             viewer(prob, pt)
#         if i == 0:
#             dump_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp.pkl')

#     load_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp.pkl')

T4mat = np.zeros((7, 7, 9))
thrustmat = np.zeros((7, 7, 9))
Wfmat = np.zeros((7, 7, 9))
SMNmat = np.zeros((7, 7, 9))
SMWmat = np.zeros((7, 7, 9))

with open('fn.pkl', 'rb') as fh:
    fullfn = pickle.load(fh)

thrustmat = np.load('thrust0W_corr2.npy')
T4mat = np.load('T40W_corr2.npy')
Wfmat = np.load('Wf0W_corr2.npy')
SMNmat = np.load('SMN0W_corr2.npy')
SMWmat = np.load('SMW0W_corr2.npy')

altlist = [35000., 30000., 25000., 20000., 15000., 10000., 5000.]
mlist = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
throtlist = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.4, 0.5, 0.7, 0.9]
throtreorder = [0, 2, 4, 6, 7, 8, 6, 5, 3, 1]
# throtlist = [0.5, 0.4]
# throtreorder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
# altlist = [30000]
# mlist = [0.5]
# throtlist = [0.5]
# throtreorder = [0, 1, 2, 3, 4, 5]

# for powerlevel in [0.2, 0.5]:
#     print('========Stepping up to '+str(powerlevel)+'==============')
#     prob['CRZ.motor.power'] = powerlevel
#     prob.run_model()
#     for pt in pts:
#         viewer(prob, pt)

# for throttle in [1.0, 0.8, 0.6, 0.5, 0.45, 0.4]:
#     print('========Stepping up to '+str(throttle)+'==============')
#     prob.run_model()
#     for pt in pts:
#         viewer(prob, pt)
prob['CRZ.motor.power'] = 0.0
prob.run_model()
dump_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp.pkl')
try:
    for i, alt in enumerate(altlist):
        for j, MN in enumerate(mlist):
            prob['CRZ:MN'] = MN
            prob['CRZ:alt'] = alt
            fullthrust = fullfn[i][j]
            for kfake, throttle in enumerate(throtlist):
                k = throtreorder[kfake]
                fntarget = fullthrust * throttle
                print('======Running MN ' +str(MN)+' alt '+str(alt)+' throttle '+str(throttle)+'===================')
                prob['CRZ:Fn_target'] = fntarget
                try:
                    if thrustmat[i,j,k] <= 0.0:
                        prob.run_model()
                        if j == 0:
                            dump_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp.pkl')
                        if k == 0:
                            dump_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp_throt.pkl')
                        for pt in pts:
                            viewer(prob, pt)
                        thrustmat[i,j,k] = prob['CRZ.perf.Fn'][0]
                        T4mat[i,j,k] = prob['CRZ.burner.Fl_O:tot:T'][0]
                        Wfmat[i,j,k] = prob['CRZ.burner.Wfuel'][0]
                        SMNmat[i,j,k] = prob['CRZ.lpc.SMN'][0]
                        SMWmat[i,j,k] = prob['CRZ.lpc.SMW'][0]
                        np.save('thrust0W_corr2', thrustmat)
                        np.save('T40W_corr2', T4mat)
                        np.save('Wf0W_corr2', Wfmat)
                        np.save('SMN0W_corr2', SMNmat)
                        np.save('SMW0W_corr2', SMWmat)
                except Exception as e:
                    print(e)
                    thrustmat[i,j,k] = -1
                    T4mat[i,j,k] = -1
                    Wfmat[i,j,k] = -1 
                    SMNmat[i,j,k] = -1
                    SMWmat[i,j,k] = -1
            # load_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp_throt.pkl')
        # load_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp.pkl')
except Exception as e:
    print(e)




#M8 0.4 
# 2504.9050813658396
# 2241.991117916585
# 0.27331403453982894

#M6 0.7
# 4383.583892390225
# 2401.3318774909085
# 0.36536724141602644
#M3 0.4
#M2 0.4

# 30k 0.8 0.4
# 2504.9050813656104
# 2311.820183974654
# 0.34210032099690446

# 30k M0.8 0.2
# 1252.4525406816356
# 2071.70061826198
# 0.20691404466809216

# 30k M0.6 0.3
# 1878.6788110243142
# 2057.1343784468254
# 0.18883912707820436

# 30k M0.5 0.5 
# 3131.1313517073067
# 2241.964336510856
# 0.2849249708178731

# 30k M0.5 0.4
# 2504.9050813658214
# 2159.396704878729
# 0.23799407360100547
# prob.model.list_outputs(explicit=True, residuals=True, residuals_tol=1e-6)

# prob.check_totals(of=['TOC.perf.Fn','RTO.perf.Fn','SLS.perf.Fn','CRZ.perf.Fn',
#                                 'TOC.perf.TSFC','RTO.perf.TSFC','SLS.perf.TSFC','CRZ.perf.TSFC',
#                                 'TOC.fan_dia.FanDia', 'bal.TOC_BPR','TOC.hpc_CS.CS',],
#                                 wrt=['OPR', 'TR', 'fan:PRdes', 'lpc:PRdes', 'RTO:T4max', 'CRZ:VjetRatio'],
#                                 step_calc='rel', step=1e-5)

# data = prob.compute_totals(of=['TOC.perf.Fn','RTO.perf.Fn','SLS.perf.Fn','CRZ.perf.Fn',
#                                 'TOC.perf.TSFC','RTO.perf.TSFC','SLS.perf.TSFC','CRZ.perf.TSFC',
#                                 'TOC.fan_dia.FanDia', 'bal.TOC_BPR','TOC.hpc_CS.CS',], wrt=['OPR_simple', 'TR', 'fan:PRdes', 'lpc:PRdes', 'RTO:T4max', 'CRZ:VjetRatio'])
# pprint(data)


# prob.check_partials(comps=['TOC'], compact_print=True)
