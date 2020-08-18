# --- Python 3.8 ---
# FileName: N3_v1_pressure_sweep.py
# Created by: alamkin
# Date: 7/16/20
# Last Updated: 10:37 AM

# --- Imports ---
import os
import time
import pickle
import openmdao.api as om
from N3ref_v1 import N3, viewer, dump_guess, load_guess

def run_model(pressure_drop, data_fp = None, record=False):
    """
    Encapsulated method for running the N+3 engine model
    Parameters
    ----------
    pressure_drop: float, dPqP in the bypass duct
    data_fp: string, filepath for where to save the pressure sweep data
    record: bool, default = False, Set to true in order to record model data using sql

    Returns
    -------

    """


    prob = om.Problem()

    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

    des_vars.add_output('inlet:ram_recovery', 0.9980),
    des_vars.add_output('fan:PRdes', 1.30444717),
    des_vars.add_output('fan:effDes', 0.96888),
    des_vars.add_output('fandiam', 100., units='inch')
    des_vars.add_output('fan:effPoly', 0.97),
    des_vars.add_output('duct2:dPqP', 0.0100),
    des_vars.add_output('lpc:PRdes', 4.000),
    des_vars.add_output('lpc:effDes', 0.889513),
    des_vars.add_output('lpc:effPoly', 0.905),
    des_vars.add_output('duct25:dPqP', 0.0150),
    des_vars.add_output('hpc:PRdes', 14.103),
    des_vars.add_output('OPR_simple', 64.49032463)
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
    des_vars.add_output('duct17:dPqP', pressure_drop),
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
    des_vars.add_output('bld3:bld_inlet:frac_W', 0.05349481),  # different than NPSS due to Wref
    des_vars.add_output('bld3:bld_exit:frac_W', 0.04701518),  # different than NPSS due to Wref
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
    des_vars.add_output('heat_engine:Wdot', 0.0001, units='kW')
    des_vars.add_output('heat_engine:eff_factor', 0.4)

    # POINT 1: Top-of-climb (TOC)
    des_vars.add_output('TOC:alt', 35000., units='ft'),
    des_vars.add_output('TOC:MN', 0.8),
    des_vars.add_output('TOC:T4max', 3150.0, units='degR'),
    des_vars.add_output('TOC:Fn_des', 6123.0, units='lbf'),
    des_vars.add_output('TOC:ram_recovery', 0.9980),
    des_vars.add_output('TOC:BPR', 20.31168463, units=None)
    des_vars.add_output('TOC:W', 810.91768342, units='lbm/s')
    des_vars.add_output('TR', 0.926470588)



    # POINT 4: Cruise (CRZ)
    des_vars.add_output('CRZ:MN', 0.8),
    des_vars.add_output('CRZ:alt', 35000.0, units='ft'),
    des_vars.add_output('CRZ:Fn_target', 5510.7284981, units='lbf'),
    des_vars.add_output('CRZ:dTs', 0.0, units='degR')
    des_vars.add_output('CRZ:Ath', 4747.1, units='inch**2')
    des_vars.add_output('CRZ:RlineMap', 1.9397)
    des_vars.add_output('CRZ:ram_recovery', 0.9980),
    des_vars.add_output('CRZ:duct2:dPqP', 0.0092)
    des_vars.add_output('CRZ:duct25:dPqP', 0.0138)
    des_vars.add_output('CRZ:duct45:dPqP', 0.0050)
    des_vars.add_output('CRZ:duct5:dPqP', 0.0097)
    des_vars.add_output('CRZ:VjetRatio', 1.35)
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
    prob.model.connect('heat_engine:Wdot', 'TOC.heat_engine.Wdot')
    prob.model.connect('heat_engine:eff_factor', 'TOC.heat_engine.eff_factor')

    # OTHER POINTS (OFF-DESIGN)
    pts = ['CRZ']

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
        prob.model.connect('heat_engine:Wdot', pt+'.heat_engine.Wdot')
        prob.model.connect('heat_engine:eff_factor', pt+'.heat_engine.eff_factor')

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
    prob.model.connect('bld3:bld_exit:frac_W', 'CRZ.bld3.bld_exit:frac_W')
    prob.model.connect('bld3:bld_inlet:frac_W', 'CRZ.bld3.bld_inlet:frac_W')

    bal = prob.model.add_subsystem('bal', om.BalanceComp())
    # bal.add_balance('TOC_BPR', val=20.7281, units=None, eq_units='ft/s', use_mult=True)
    # prob.model.connect('bal.TOC_BPR', 'TOC.splitter.BPR')
    # prob.model.connect('CRZ.byp_nozz.Fl_O:stat:V', 'bal.lhs:TOC_BPR')
    # prob.model.connect('CRZ.core_nozz.Fl_O:stat:V', 'bal.rhs:TOC_BPR')
    # prob.model.connect('CRZ:VjetRatio', 'bal.mult:TOC_BPR')
    prob.model.connect('TOC:BPR', 'TOC.splitter.BPR')


    # bal.add_balance('TOC_W', val=810.918, units='lbm/s', eq_units='inch')
    # prob.model.connect('bal.TOC_W', 'TOC.fc.W')
    # prob.model.connect('TOC.fan_dia.FanDia', 'bal.lhs:TOC_W')
    # prob.model.connect('fandiam', 'bal.rhs:TOC_W')
    # prob.model.connect('RTO.burner.Fl_O:tot:T', 'bal.lhs:TOC_W')
    # prob.model.connect('RTO:T4max','bal.rhs:TOC_W')

    # bal.add_balance('CRZ_Fn_target', val=5514.4, units='lbf', eq_units='lbf', use_mult=True, mult_val=0.9, ref0=5000.0, ref=7000.0)
    # prob.model.connect('bal.CRZ_Fn_target', 'CRZ.balance.rhs:FAR')
    # prob.model.connect('TOC.perf.Fn', 'bal.lhs:CRZ_Fn_target')
    # prob.model.connect('CRZ.perf.Fn','bal.rhs:CRZ_Fn_target')

    prob.model.connect('TOC:W', 'TOC.fc.W')
    prob.model.connect('TOC:Fn_des', 'TOC.balance.rhs:FAR')

    prob.model.add_subsystem('eng_op_limits',
                            om.ExecComp('lim_ratio = max(T4/T4max, N1/N1max, N2/N2max)',
                                        lim_ratio={'value': 0.95, 'units':None},
                                        T4={'value': 2800., 'units':'degR'},
                                        T4max={'value': 3040, 'units': 'degR'}, #3040
                                        N1={'value': 6600., 'units': 'rpm'},
                                        N1max={'value': 6800., 'units': 'rpm'},
                                        N2={'value': 20000., 'units': 'rpm'},
                                        N2max={'value': 22500., 'units': 'rpm'}))
    prob.model.connect('CRZ.burner.Fl_O:tot:T', 'eng_op_limits.T4')
    prob.model.connect('CRZ.balance.lp_Nmech', 'eng_op_limits.N1')
    prob.model.connect('CRZ.balance.hp_Nmech', 'eng_op_limits.N2')
    # prob.model.connect('CRZ:Fn_target', 'CRZ.balance.rhs:FAR')


    bal.add_balance('CRZ_Fn_target', val=6200., units='lbf', eq_units=None, rhs_val=1.0, ref0=5000.0, ref=7000.0)
    prob.model.connect('bal.CRZ_Fn_target', 'CRZ.balance.rhs:FAR')
    prob.model.connect('eng_op_limits.lim_ratio', 'bal.lhs:CRZ_Fn_target')

    prob.model.set_order(['des_vars', 'TOC', 'CRZ', 'eng_op_limits','bal'])

    newton = prob.model.nonlinear_solver = om.NewtonSolver()
    newton.options['atol'] = 1e-10
    newton.options['rtol'] = 1e-10
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    newton.options['solve_subsystems'] = True
    newton.options['max_sub_solves'] = 10
    newton.options['err_on_non_converge'] = True
    newton.options['reraise_child_analysiserror'] = False
    newton.linesearch = om.BoundsEnforceLS()
    newton.linesearch.options['bound_enforcement'] = 'scalar'
    newton.linesearch.options['iprint'] = -1

    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

    # setup the optimization
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
    # prob.driver.opt_settings = {'Major step limit': 0.05}

    prob.model.add_design_var('fan:PRdes', lower=1.30, upper=1.4)
    prob.model.add_design_var('lpc:PRdes', lower=2.5, upper=4.0)
    prob.model.add_design_var('OPR_simple', lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
    # prob.model.add_design_var('RTO:T4max', lower=3000.0, upper=3400.0, ref0=3000.0, ref=3400.0)
    prob.model.add_design_var('CRZ:VjetRatio', lower=1.35, upper=1.45, ref0=1.35, ref=1.45)
    # prob.model.add_design_var('TR', lower=0.5, upper=0.95, ref0=0.5, ref=0.95)
    prob.model.add_design_var('fandiam', lower=99.0, upper=100.0)
    # prob.model.add_objective('TOC.perf.TSFC')
    prob.model.add_objective('CRZ.burner.Wfuel', ref0=.6, ref=0.7)
    # to add the constraint to the model
    # prob.model.add_constraint('TOC.fan_dia.FanDia', upper=101.0, ref=100.0)
    # prob.model.add_constraint('TOC.perf.Fn', lower=5800.0, ref=6000.0)
    prob.model.add_constraint('TOC.burner.Fl_O:tot:T', upper=3400.*0.95)
    prob.setup(check=False)

    # prob['RTO.hpt_cooling.x_factor'] = 0.9

    # initial guesses
    prob['TOC.balance.FAR'] = 0.02650
    # prob['bal.TOC_W'] = 810.95
    prob['TOC.balance.lpt_PR'] = 10.937
    prob['TOC.balance.hpt_PR'] = 4.185
    prob['TOC.fc.balance.Pt'] = 5.272
    prob['TOC.fc.balance.Tt'] = 444.41

    # for pt in ['RTO']:
    #     prob.set_val(pt+'.motor.power', 1000., units='kW')

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

    if record:
        # double check that the path exists
        if os.path.exists(data_fp):
            file = str(pressure_drop)
            file = file.replace('.','')
            prob_rec = om.SqliteRecorder(data_fp + "/pressureSweep_" + file + ".sql")
        else:
            print('File directory not properly set up')
        prob.add_recorder(prob_rec)

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()
    try:
        load_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'initpklod.pkl')
    except FileNotFoundError:
        prob.run_model()
        dump_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'initpklod.pkl')

    try:
        with open('t4.pkl', 'rb') as fh:
            T4outer = pickle.load(fh)
        with open('fn.pkl', 'rb') as fh:
            Touter = pickle.load(fh)
        with open('wf.pkl', 'rb') as fh:
            Wfouter = pickle.load(fh)
    except FileNotFoundError:
        print('Starting from scratch')
        T4outer = []
        Touter = []
        Wfouter = []

    for alt in [35000., 30000., 25000., 20000., 15000., 10000., 5000., 0.]:
    # for alt in [0.]:
        T4inner = []
        Tinner = []
        Wfinner = []
        for i, MN in enumerate([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]):
            # separate rules for zero altitude
            if alt < 4999.:
                # can run hotter here because time limited TO thrust
                prob['eng_op_limits.T4max'] = 3150
                if MN > 0.6:
                    # M > 0.6 fails to converge at SL
                    print('Skipping excessive SL mach')
                    Tinner.append(-1)
                    T4inner.append(-1)
                    Wfinner.append(-1)
                    continue
            print('======Running MN ' +str(MN)+' alt '+str(alt)+'===================')
            prob['CRZ:MN'] = MN
            prob['CRZ:alt'] = alt
            prob.run_model()
            for pt in pts:
                viewer(prob, pt)
            Tinner.append(prob['CRZ.perf.Fn'].copy())
            T4inner.append(prob['CRZ.burner.Fl_O:tot:T'].copy())
            Wfinner.append(prob['CRZ.burner.Wfuel'].copy())
            if i == 0:
                dump_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp.pkl', OTHER_PARMS=['bal.CRZ_Fn_target'])
            if record:
                prob.record(case_name='mach_'+str(MN)+'_alt_'+str(alt))
        T4outer.append(T4inner)
        Touter.append(Tinner)
        Wfouter.append(Wfinner)
        load_guess(prob, DPARMS, ODPARMS, DNAME, ODNAMES, 'sweep_tmp.pkl', OTHER_PARMS=['bal.CRZ_Fn_target'])
        
    with open('t4.pkl', 'wb') as fh:
        pickle.dump(T4outer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open('fn.pkl', 'wb') as fh:
        pickle.dump(Touter, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open('wf.pkl', 'wb') as fh:
        pickle.dump(Wfouter, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if not record:
        exit()


def pressure_sweep():

    # check for proper data directory and create one if not already made
    fp = "../../../"
    if not os.path.exists(fp+"max_throttle_sweeps"):
        os.makedirs(fp+"max_throttle_sweeps/pressure_sweeps/raw")
    elif not os.path.exists(fp+"max_throttle_sweeps/pressure_sweeps"):
        os.makedirs(fp+"max_throttle_sweeps/pressure_sweeps/raw")
    elif not os.path.exists(fp+"max_throttle_sweeps/pressure_sweeps/raw"):
        os.makedirs(fp+"max_throttle_sweeps/pressure_sweeps/raw")

    full_fp = fp+"max_throttle_sweeps/pressure_sweeps/raw"

    dir = os.listdir(full_fp)
    dir = dir[1:]
    if len(dir) != 0:
        user_in = input('Data already exists.  Overwrite old files? [y/n]: ')
        if user_in.lower() == 'y':
            for file in dir:
                os.remove(os.path.join(full_fp, file))
                print("Removed {} from {}".format(file, full_fp))
        else:
            print('Sweep cancelled.')
            return

    pressure_sweep = [0.015]  # dPqP
    for val in pressure_sweep:
        run_model(val, data_fp=full_fp, record=True)

if __name__=="__main__":
    pressure_sweep()