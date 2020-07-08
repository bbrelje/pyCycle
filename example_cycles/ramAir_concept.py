# --- Python 3.8 ---
# FileName: ramAir_concept.py
# Created by: alamkin
# Date: 7/6/20
# Last Updated: 11:05 AM

# --- Imports ---
import sys
import numpy as np

import openmdao.api as om

import pycycle.api as pyc

class RamAir(om.Group):
    """
    Ram air duct concept cycle to demonstrate the heat transfer mechanics for the N+3 engine updated bypass design
    """
    def initialize(self):
        """
        Initializes the options for this group
        :return:
        """
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')
    def setup(self):
        """
        Sets up the ram-air model following the steps listed below:

        1) Specifies the thermal data for the jet-fuel
        2) Adds subsystems to the model
        3) Connects the flow stations
        4) Connects the performance elements
        5) Connects the heat transfer elements
        6) Specifies the solver order
        7) Adds balances to the model
        8) Implements the solvers
        """
        # --- Specify the thermal properties for the jet-fuel used ---
        thermo_spec = pyc.species_data.janaf

        # --- Specify the design case from the options dict ---
        design = self.options['design']

        # --- Add subsystems ---
        # Heat Transfer
        self.add_subsystem('heat_transfer', om.ExecComp(['q_in=tms_q'],
                                                        q_in={'units':'Btu/s'}),
                           promotes_inputs=['tms_q'])
        # Inlet
        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))

        # Inlet to nozzle duct
        self.add_subsystem('duct', pyc.Duct(design=design,thermo_data=thermo_spec, elements=pyc.AIR_MIX))

        # Nozzle
        self.add_subsystem('nozz',
                           pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_MIX))

        # Performance
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # --- Connect flow stations ---
        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        pyc.connect_flow(self, 'inlet.Fl_O', 'duct.Fl_I')
        pyc.connect_flow(self, 'duct.Fl_O', 'nozz.Fl_I')

        # --- Connect Nozzle to freestream static conditions
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        # --- Connect properties between engine elements and the performance element ---
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        # --- Connect heat transfer ---
        self.connect('heat_transfer.q_in', 'duct.Q_dot')


        # Add balances for design
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('W', units='lbm/s', eq_units='lbf')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('perf.Fn', 'balance.lhs:W')

        # --- Setup solver to converge engine ---
        main_order = ['heat_transfer', 'fc', 'inlet', 'duct',
                      'nozz', 'perf', 'balance']
        self.set_order(main_order)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver(assemble_jac = True)

def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC  ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" % (
    prob[pt + '.fc.Fl_O:stat:MN'], prob[pt + '.fc.alt'], prob[pt + '.inlet.Fl_O:stat:W'], prob[pt + '.perf.Fn'],
    prob[pt + '.perf.Fg'], prob[pt + '.inlet.F_ram'], prob[pt + '.perf.OPR'], prob[pt + '.perf.TSFC']),
          file=file, flush=True)

    # print duct heat transfer values
    duct_names = ['duct']
    duct_full_names = [f'{pt}.{duct}' for duct in duct_names]
    pyc.print_duct_heat_transfer(prob, duct_full_names, file=file)

    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'duct.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    noz_names = ['nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

if __name__ == "__main__":
    import time
    from openmdao.api import Problem, IndepVarComp
    from openmdao.utils.units import convert_units as cu

    prob = om.Problem()

    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

    des_vars.add_output('alt', 35000.0, units='ft'),
    des_vars.add_output('MN', 0.8),
    des_vars.add_output('Fn_des', 300.0, units='lbf'),
    des_vars.add_output('duct:dPqP', 0.02),
    des_vars.add_output('nozz:Cv', 0.99),
    des_vars.add_output('inlet:MN_out', 0.6),
    des_vars.add_output('tms_q')

    # Create design instance of model
    prob.model.add_subsystem('DESIGN', RamAir())

    # Connect design point inputs to model
    prob.model.connect('alt', 'DESIGN.fc.alt')
    prob.model.connect('MN', 'DESIGN.fc.MN')
    prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')

    prob.model.connect('duct:dPqP', 'DESIGN.duct.dPqP')
    prob.model.connect('nozz:Cv', 'DESIGN.nozz.Cv')

    prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')

    prob.model.connect('tms_q', 'DESIGN.tms_q')

    prob.setup(check=False)

    # Set initial guesses for balances
    prob['DESIGN.balance.W'] = 168.453135137
    prob['DESIGN.fc.balance.Pt'] = 14.6955113159
    prob['DESIGN.fc.balance.Tt'] = 518.665288153

    # set the thermal management system value for heat in and out of ducts
    prob['tms_q'] = 10000.0  # units: Btu/s

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    viewer(prob, 'DESIGN')

    print()
    print("time", time.time() - st)
