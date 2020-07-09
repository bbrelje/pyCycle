# --- Python 3.8 ---
# FileName: thermal.py
# Created by: alamkin
# Date: 7/9/20
# Last Updated: 7:20 PM

# --- Imports ---
from __future__ import division
from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, NewtonSolver, DirectSolver, BoundsEnforceLS
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent

import numpy as np
import scipy.sparse as sp
import sys, os

sys.path.insert(0, os.getcwd())

"""Analysis routines for simulating thermal management of aircraft components"""


class SimpleEngine(ExplicitComponent):
    """
    Convert heat to work based on an assumed fraction of the
    Carnot efficiency.
    Inputs
    ------
    T_h : float
        Temperature of the hot heat input (vector, K)
    T_c : float
        Temperature of the cold heat input (vector, K)
    Wdot : float
        Work generation rate (vector, W)
    eff_factor : float
        Percentage of the Carnot efficiency (scalar, dimensionless)
    Outputs
    -------
    q_h : float
        Heat extracted from the hot side (vector, W)
    q_c : float
        Waste heat sent to cold side (vector, W)
    eta_thermal : float
        Overall thermal efficiency (vector, dimensionless)
    Options
    -------
    num_nodes : float
        The number of analysis points to run
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('T_h', units='K', shape=(nn_tot,), val=600.)
        self.add_input('T_c', units='K', shape=(nn_tot,), val=400.)
        self.add_input('Wdot', units='W', shape=(nn_tot,), val=1000.)
        self.add_input('eff_factor', units=None, val=0.4)

        self.add_output('q_h', units='W', shape=(nn_tot,))
        self.add_output('q_c', units='W', shape=(nn_tot,))
        self.add_output('eta_thermal', units=None, shape=(nn_tot,))

        self.declare_partials(['q_h'], ['T_h', 'T_c', 'Wdot'], rows=arange, cols=arange)
        self.declare_partials(['q_c'], ['T_h', 'T_c', 'Wdot'], rows=arange, cols=arange)
        self.declare_partials(['eta_thermal'], ['T_h', 'T_c'], rows=arange, cols=arange)

        self.declare_partials(['q_h'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
        self.declare_partials(['q_c'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
        self.declare_partials(['eta_thermal'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))

    def compute(self, inputs, outputs):
        # compute carnot efficiency
        # 1 - Tc/Th
        eta_carnot = 1 - inputs['T_c'] / inputs['T_h']
        eta_thermal = inputs['eff_factor'] * eta_carnot
        outputs['eta_thermal'] = eta_thermal
        # compute the heats
        outputs['q_h'] = inputs['Wdot'] / eta_thermal
        outputs['q_c'] = inputs['Wdot'] / eta_thermal - inputs['Wdot']

    def compute_partials(self, inputs, J):
        eta_carnot = 1 - inputs['T_c'] / inputs['T_h']
        eta_thermal = inputs['eff_factor'] * eta_carnot

        J['eta_thermal', 'T_h'] = inputs['eff_factor'] * inputs['T_c'] / inputs['T_h'] ** 2
        J['eta_thermal', 'T_c'] = inputs['eff_factor'] * (-1 / inputs['T_h'])
        J['eta_thermal', 'eff_factor'] = eta_carnot

        J['q_h', 'T_h'] = - inputs['Wdot'] / eta_thermal ** 2 * (
                    inputs['eff_factor'] * inputs['T_c'] / inputs['T_h'] ** 2)
        J['q_h', 'T_c'] = - inputs['Wdot'] / eta_thermal ** 2 * (inputs['eff_factor'] * (-1 / inputs['T_h']))
        J['q_h', 'Wdot'] = 1 / eta_thermal
        J['q_h', 'eff_factor'] = - inputs['Wdot'] / eta_thermal ** 2 * (eta_carnot)

        J['q_c', 'T_h'] = - inputs['Wdot'] / eta_thermal ** 2 * (
                    inputs['eff_factor'] * inputs['T_c'] / inputs['T_h'] ** 2)
        J['q_c', 'T_c'] = - inputs['Wdot'] / eta_thermal ** 2 * (inputs['eff_factor'] * (-1 / inputs['T_h']))
        J['q_c', 'Wdot'] = (1 / eta_thermal - 1)
        J['q_c', 'eff_factor'] = - inputs['Wdot'] / eta_thermal ** 2 * (eta_carnot)


class SimpleHeatPump(ExplicitComponent):
    """
    Pumps heat from cold source to hot sink with work input
    based on assumed fraction of Carnot efficiency.
    Inputs
    ------
    T_h : float
        Temperature of the hot heat input (vector, K)
    T_c : float
        Temperature of the cold heat input (vector, K)
    Wdot : float
        Work usage rate (vector, W)
    eff_factor : float
        Percentage of the Carnot efficiency (scalar, dimensionless)

    Outputs
    -------
    q_c : float
        Heat extracted from the cold side (vector, W)
    q_h : float
        Heat sent to hot side (vector, W)
    COP_cooling : float
        Cooling coefficient of performance, heat removed from cold side
        divided by work used (vector, dimensionless)
    Options
    -------
    num_nodes : float
        The number of analysis points to run
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points to run')

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('T_h', units='K', shape=(nn_tot,), val=600.)
        self.add_input('T_c', units='K', shape=(nn_tot,), val=400.)
        self.add_input('Wdot', units='W', shape=(nn_tot,), val=1000.)
        self.add_input('eff_factor', units=None, val=0.4)

        self.add_output('q_c', units='W', shape=(nn_tot,))
        self.add_output('q_h', units='W', shape=(nn_tot,))
        self.add_output('COP_cooling', units=None, shape=(nn_tot,))

        self.declare_partials(['q_c'], ['T_h', 'T_c', 'Wdot'], rows=arange, cols=arange)
        self.declare_partials(['q_h'], ['T_h', 'T_c', 'Wdot'], rows=arange, cols=arange)
        self.declare_partials(['COP_cooling'], ['T_h', 'T_c'], rows=arange, cols=arange)

        self.declare_partials(['q_c'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
        self.declare_partials(['q_h'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
        self.declare_partials(['COP_cooling'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))

    def compute(self, inputs, outputs):
        # Cooling coefficient of performance
        COP_cooling = inputs['eff_factor'] * inputs['T_c'] / (inputs['T_h'] - inputs['T_c'])
        outputs['COP_cooling'] = COP_cooling

        # Heat transfer
        outputs['q_c'] = COP_cooling * inputs['Wdot']
        outputs['q_h'] = (1 + COP_cooling) * inputs['Wdot']

    def compute_partials(self, inputs, J):
        # Assign inputs to variables for readability
        T_h = inputs['T_h']
        T_c = inputs['T_c']
        Wdot = inputs['Wdot']
        eff_factor = inputs['eff_factor']

        J['COP_cooling', 'T_h'] = - eff_factor * T_c / (T_h - T_c) ** 2
        J['COP_cooling', 'T_c'] = eff_factor * T_h / (T_h - T_c) ** 2
        J['COP_cooling', 'eff_factor'] = T_c / (T_h - T_c)

        J['q_c', 'T_h'] = - eff_factor * Wdot * T_c / (T_h - T_c) ** 2
        J['q_c', 'T_c'] = eff_factor * Wdot * T_h / (T_h - T_c) ** 2
        J['q_c', 'Wdot'] = eff_factor * T_c / (T_h - T_c)
        J['q_c', 'eff_factor'] = Wdot * T_c / (T_h - T_c)

        J['q_h', 'T_h'] = - eff_factor * Wdot * T_c / (T_h - T_c) ** 2
        J['q_h', 'T_c'] = eff_factor * Wdot * T_h / (T_h - T_c) ** 2
        J['q_h', 'Wdot'] = 1 + eff_factor * T_c / (T_h - T_c)
        J['q_h', 'eff_factor'] = Wdot * T_c / (T_h - T_c)


class SimpleTMS(Group):
    """
    Models a simple thermal management system cooling an electric motor
    with a heat pump (refrigerator). The refrigerator is balanced to consume
    an amount of power such that all the heat out of the motor is removed by
    the refrigerator.
    Inputs
    ------
    throttle : float
        Motor power control setting. Should be [0, 1]. (vector, dimensionless)
    motor_elec_power_rating: float
        Motor electric (not mech) design power. (scalar, W)

    Outputs
    -------
    shaft_power_out : float
        Shaft power output from motor (vector, W)
    motor_elec_load : float
        Electrical load consumed by motor (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_weight : float
        Weight of the component (scalar, kg)
    component_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)
    q_h : float
        Heat sent to hot side of refrigerator (vector, W)
    COP_cooling : float
        Cooling coefficient of performance of refrigerator, heat removed from motor
        divided by work used (vector, dimensionless)
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    motor_efficiency : float
        Motor shaft power efficiency. Sensible range 0.0 to 1.0 (default 0.93, same as magni500)
    motor_weight_inc : float
        Motor weight per unit rated power (default 2.411e-4, same as magni500, kg/W)
    motor_weight_base : float
        Motor base weight (default 0, kg)
    motor_cost_inc : float
        Motor cost per unit rated power (default 0.134228, USD/W)
    motor_cost_base : float
        Motor base cost (default 1 USD) B
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points to run')
        self.options.declare('motor_efficiency', default=0.93, desc='Motor efficiency (dimensionless)')
        self.options.declare('motor_weight_inc', default=135. / 560e3, desc='Motor kg/W')
        self.options.declare('motor_weight_base', default=0., desc='Motor base weight kg')
        self.options.declare('motor_cost_inc', default=100. / 745., desc='Motor cost per watt $/W')
        self.options.declare('motor_cost_base', default=1., desc='Motor cost base $')

    def setup(self):
        nn = self.options['num_nodes']
        # Add the electric motor, heat pump, heat balance, and motor/heat sink temps to the Group
        self.add_subsystem('motor', SimpleMotor(num_nodes=nn,
                                                efficiency=self.options['motor_efficiency'],
                                                weight_inc=self.options['motor_weight_inc'],
                                                weight_base=self.options['motor_weight_base'],
                                                cost_inc=self.options['motor_cost_inc'],
                                                cost_base=self.options['motor_cost_base']),
                           promotes_inputs=['throttle', ('elec_power_rating', 'motor_elec_power_rating')],
                           promotes_outputs=['shaft_power_out', ('elec_load', 'motor_elec_load'), 'component_cost',
                                             'component_weight', 'component_sizing_margin'])

        self.add_subsystem('refrigerator', SimpleHeatPump(num_nodes=nn),
                           promotes_outputs=['q_h', 'COP_cooling'])

        self.add_subsystem('heat_bal', BalanceComp(name='Wdot', units='W', eq_units='W', val=np.ones(nn),
                                                   lhs_name='refrigerator_heat_in', rhs_name='motor_heat_out'))

        temps = self.add_subsystem('temps', IndepVarComp())
        temps.add_output('motor', val=20., shape=(nn,), units='degC')
        temps.add_output('heat_sink', val=100., shape=(nn,), units='degC')

        # Connect the electric motor and refrigerator to the heat balance so it solves for
        # refrigerator work required to remove heat generated by motor
        self.connect('motor.heat_out', 'heat_bal.motor_heat_out')
        self.connect('refrigerator.q_c', 'heat_bal.refrigerator_heat_in')
        self.connect('heat_bal.Wdot', 'refrigerator.Wdot')

        # Connect the motor and heat sink temperatures to the refrigerator cold and hot sides, respectively
        self.connect('temps.motor', 'refrigerator.T_c')
        self.connect('temps.heat_sink', 'refrigerator.T_h')


class ThermalComponentWithMass(ExplicitComponent):
    """
    Computes thermal residual of a component with heating, cooling, and thermal mass
    Inputs
    ------
    q_in : float
        Heat generated by the component (vector, W)
    q_out : float
        Heat to waste stream (vector, W)
    mass : float
        Thermal mass (scalar, kg)
    Outputs
    -------
    dTdt : float
        First derivative of temperature (vector, K/s)
    Options
    -------
    specific_heat : float
        Specific heat capacity of the object in J / kg / K (default 921 = aluminum)
    num_nodes : float
        The number of analysis points to run
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('specific_heat', default=921, desc='Specific heat in J/kg/K - default 921 for aluminum')

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('q_in', units='W', shape=(nn_tot,))
        self.add_input('q_out', units='W', shape=(nn_tot,))
        self.add_input('mass', units='kg')
        self.add_output('dTdt', units='K/s', shape=(nn_tot,))

        self.declare_partials(['dTdt'], ['q_in'], rows=arange, cols=arange)
        self.declare_partials(['dTdt'], ['q_out'], rows=arange, cols=arange)
        self.declare_partials(['dTdt'], ['mass'], rows=arange, cols=np.zeros((nn_tot,)))

    def compute(self, inputs, outputs):
        spec_heat = self.options['specific_heat']
        outputs['dTdt'] = (inputs['q_in'] - inputs['q_out']) / inputs['mass'] / spec_heat

    def compute_partials(self, inputs, J):
        nn_tot = self.options['num_nodes']
        spec_heat = self.options['specific_heat']

        J['dTdt', 'mass'] = - (inputs['q_in'] - inputs['q_out']) / inputs['mass'] ** 2 / spec_heat
        J['dTdt', 'q_in'] = 1 / inputs['mass'] / spec_heat
        J['dTdt', 'q_out'] = - 1 / inputs['mass'] / spec_heat


class CoolantReservoirRate(ExplicitComponent):
    """
    Computes dT/dt of a coolant reservoir based on inflow and current temps and flow rate
    Inputs
    ------
    T_in : float
        Coolant stream in (vector, K)
    T_out : float
        Temperature of the reservoir (vector, K)
    mass : float
        Total quantity of coolant (scalar, kg)
    mdot_coolant : float
        Mass flow rate of the coolant (vector, kg/s)
    Outputs
    -------
    dTdt : float
        First derivative of temperature (vector, K/s)
    Options
    -------
    num_nodes : float
        The number of analysis points to run
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('T_in', units='K', shape=(nn_tot,))
        self.add_input('T_out', units='K', shape=(nn_tot,))
        self.add_input('mdot_coolant', units='kg/s', shape=(nn_tot,))
        self.add_input('mass', units='kg')
        self.add_output('dTdt', units='K/s', shape=(nn_tot,))

        self.declare_partials(['dTdt'], ['T_in', 'T_out', 'mdot_coolant'], rows=arange, cols=arange)
        self.declare_partials(['dTdt'], ['mass'], rows=arange, cols=np.zeros((nn_tot,)))

    def compute(self, inputs, outputs):
        outputs['dTdt'] = inputs['mdot_coolant'] / inputs['mass'] * (inputs['T_in'] - inputs['T_out'])

    def compute_partials(self, inputs, J):
        J['dTdt', 'mass'] = - inputs['mdot_coolant'] / inputs['mass'] ** 2 * (inputs['T_in'] - inputs['T_out'])
        J['dTdt', 'mdot_coolant'] = 1 / inputs['mass'] * (inputs['T_in'] - inputs['T_out'])
        J['dTdt', 'T_in'] = inputs['mdot_coolant'] / inputs['mass']
        J['dTdt', 'T_out'] = - inputs['mdot_coolant'] / inputs['mass']


class ThermalComponentMassless(ImplicitComponent):
    """
    Computes thermal residual of a component with heating, cooling, and thermal mass
    Inputs
    ------
    q_in : float
        Heat generated by the component (vector, W)
    q_out : float
        Heat to waste stream (vector, W)
    Outputs
    -------
    T_object : float
        Object temperature (vector, K/s)
    Options
    -------
    num_nodes : float
        The number of analysis points to run
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('q_in', units='W', shape=(nn_tot,))
        self.add_input('q_out', units='W', shape=(nn_tot,))
        self.add_output('T_object', units='K', shape=(nn_tot,))

        self.declare_partials(['T_object'], ['q_in'], rows=arange, cols=arange, val=np.ones((nn_tot,)))
        self.declare_partials(['T_object'], ['q_out'], rows=arange, cols=arange, val=-np.ones((nn_tot,)))

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['T_object'] = inputs['q_in'] - inputs['q_out']


class ConstantSurfaceTemperatureColdPlate_NTU(ExplicitComponent):
    """
    Computes heat rejection to fluid stream of a microchannel cold plate
    with uniform temperature
    Inputs
    ------
    T_in : float
        Coolant inlet temperature (vector, K)
    T_surface : float
        Temperature of the cold plate (vector, K)
    mdot_coolant : float
        Mass flow rate of the coolant (vector, kg/s)
    channel_length : float
        Length of each microchannel (scalar, m)
    channel_width : float
        Width of each microchannel (scalar, m)
    channel_height : float
        Height of each microchannel (scalar, m)
    n_parallel : float
        Number of fluid channels (scalar, dimensionless)
    Outputs
    -------
    q : float
        Heat transfer rate from the plate to the fluid (vector, W)
    T_out : float
        Outlet fluid temperature (vector, K)
    Options
    -------
    num_nodes : float
        The number of analysis points to run
    fluid_rho : float
        Coolant density in kg/m**3 (default 0.997, water)
    fluid_k : float
        Thermal conductivity of the fluid (W/m/K) (default 0.405, glycol/water)
    nusselt : float
        Hydraulic diameter Nusselt number of the coolant in the channels
        (default 7.54 for constant temperature infinite parallel plate)
    specific_heat : float
        Specific heat of the coolant (J/kg/K) (default 3801, glycol/water)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('fluid_rho', default=997.0, desc='Fluid density in kg/m3')
        self.options.declare('fluid_k', default=0.405, desc='Thermal conductivity of the fluid in W / mK')
        self.options.declare('nusselt', default=7.54, desc='Hydraulic diameter Nusselt number')
        self.options.declare('specific_heat', default=3801, desc='Specific heat in J/kg/K')

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('T_in', units='K', shape=(nn_tot,))
        self.add_input('T_surface', units='K', shape=(nn_tot,))
        self.add_input('channel_width', units='m')
        self.add_input('channel_height', units='m')
        self.add_input('channel_length', units='m')
        self.add_input('n_parallel')
        self.add_input('mdot_coolant', units='kg/s', shape=(nn_tot,))

        self.add_output('q', units='W', shape=(nn_tot,))
        self.add_output('T_out', units='K', shape=(nn_tot,))

        self.declare_partials(['q', 'T_out'], ['T_in', 'T_surface', 'mdot_coolant'], method='cs')
        self.declare_partials(['q', 'T_out'], ['channel_width', 'channel_height', 'channel_length', 'n_parallel'],
                              method='cs')

    def compute(self, inputs, outputs):
        Ts = inputs['T_surface']
        Ti = inputs['T_in']

        Cmin = inputs['mdot_coolant'] * self.options['specific_heat']

        # cross_section_area = inputs['channel_width'] * inputs['channel_height'] * inputs['n_parallel']
        # flow_rate = inputs['mdot_coolant'] / self.options['rho'] / cross_section_area # m/s
        surface_area = 2 * (inputs['channel_width'] * inputs['channel_length'] +
                            inputs['channel_height'] * inputs['channel_length']) * inputs['n_parallel']
        d_h = 2 * inputs['channel_width'] * inputs['channel_height'] / (
                    inputs['channel_width'] + inputs['channel_height'])

        h = self.options['nusselt'] * self.options['fluid_k'] / d_h
        ntu = surface_area * h / Cmin
        effectiveness = 1 - np.exp(-ntu)
        outputs['q'] = effectiveness * Cmin * (Ts - Ti)
        outputs['T_out'] = inputs['T_in'] + outputs['q'] / inputs['mdot_coolant'] / self.options['specific_heat']


class LiquidCooledComp(Group):
    """A component (heat producing) with thermal mass
    cooled by a cold plate.
    Inputs
    ------
    q_in : float
        Heat produced by the operating component (vector, W)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    T_in : float
        Instantaneous coolant inflow temperature (vector, K)
    mass : float
        Object mass (only required in thermal mass mode) (scalar, kg)
    T_initial : float
        Initial temperature of the cold plate (only required in thermal mass mode) / object (scalar, K)
    duration : float
        Duration of mission segment, only required in unsteady mode
    channel_width : float
        Width of coolant channels (scalar, m)
    channel_height : float
        Height of coolant channels (scalar, m)
    channel_length : float
        Length of coolant channels (scalar, m)
    n_parallel : float
        Number of identical coolant channels (scalar, dimensionless)
    Outputs
    -------
    T_out : float
        Instantaneous coolant outlet temperature (vector, K)
    T: float
        Object temperature (vector, K)
    Options
    -------
    specific_heat_object : float
        Specific heat capacity of the object in J / kg / K (default 921 = aluminum)
    specific_heat_coolant : float
        Specific heat capacity of the coolant in J / kg / K (default 3801, glycol/water)
    num_nodes : int
        Number of analysis points to run
    quasi_steady : bool
        Whether or not to treat the component as having thermal mass
    """

    def initialize(self):
        self.options.declare('specific_heat_object', default=921.0, desc='Specific heat in J/kg/K')
        self.options.declare('specific_heat_coolant', default=3801, desc='Specific heat in J/kg/K')
        self.options.declare('quasi_steady', default=False,
                             desc='Treat the component as quasi-steady or with thermal mass')
        self.options.declare('num_nodes', default=1, desc='Number of quasi-steady points to runs')

    def setup(self):
        nn = self.options['num_nodes']
        quasi_steady = self.options['quasi_steady']
        if not quasi_steady:
            self.add_subsystem('base',
                               ThermalComponentWithMass(specific_heat=self.options['specific_heat_object'],
                                                        num_nodes=nn),
                               promotes_inputs=['q_in', 'mass'])
            self.add_subsystem('integratetemp',
                               Integrator(num_intervals=int((nn - 1) / 2),
                                          quantity_units='K',
                                          diff_units='s',
                                          method='simpson',
                                          time_setup='duration'),
                               promotes_inputs=['duration', ('q_initial', 'T_initial')],
                               promotes_outputs=[('q', 'T'), ('q_final', 'T_final')])
            self.connect('base.dTdt', 'integratetemp.dqdt')
        else:
            self.add_subsystem('base',
                               ThermalComponentMassless(num_nodes=nn),
                               promotes_inputs=['q_in'],
                               promotes_outputs=['T'])
        self.add_subsystem('hex',
                           ConstantSurfaceTemperatureColdPlate_NTU(num_nodes=nn,
                                                                   specific_heat=self.options['specific_heat_coolant']),
                           promotes_inputs=['T_in', ('T_surface', 'T'), 'n_parallel', 'channel*', 'mdot_coolant'],
                           promotes_outputs=['T_out'])
        self.connect('hex.q', 'base.q_out')


class CoolantReservoir(Group):
    """A reservoir of coolant capable of buffering temperature
    Inputs
    ------
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    T_in : float
        Coolant inflow temperature (vector, K)
    mass : float
        Object mass (only required in thermal mass mode) (scalar, kg)
    T_initial : float
        Initial temperature of the coolant reservoir(only required in thermal mass mode) / object (scalar, K)
    duration : float
        Time step of each mission segment (one for each segment) (scalar, s)
        If a single segment is provided (by default) this variable will be called just 'dt'
        only required in thermal mass mode
    Outputs
    -------
    T_out : float
        Coolant outlet temperature (vector, K)
    Options
    -------
    num_nodes : int
        Number of analysis points to run
    """

    def initialize(self):
        self.options.declare('num_nodes', default=5)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('rate',
                           CoolantReservoirRate(num_nodes=nn),
                           promotes_inputs=['T_in', 'T_out', 'mass', 'mdot_coolant'])
        self.add_subsystem('integratetemp',
                           Integrator(num_intervals=int((nn - 1) / 2),
                                      quantity_units='K',
                                      diff_units='s',
                                      method='simpson',
                                      time_setup='duration'),
                           promotes_inputs=['duration', ('q_initial', 'T_initial')],
                           promotes_outputs=[('q', 'T_out'), ('q_final', 'T_final')])
        self.connect('rate.dTdt', 'integratetemp.dqdt')


class LiquidCoolantTestGroup(Group):
    """A component (heat producing) with thermal mass
    cooled by a cold plate.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=11)
        self.options.declare('quasi_steady', default=False,
                             desc='Treat the component as quasi-steady or with thermal mass')

    def setup(self):
        quasi_steady = self.options['quasi_steady']
        nn = self.options['num_nodes']

        iv = self.add_subsystem('iv', IndepVarComp(), promotes_outputs=['*'])
        # iv.add_output('q_in', val=10*np.concatenate([np.ones((nn,)),0.5*np.ones((nn,)),0.2*np.ones((nn,))]), units='kW')
        throttle_profile = np.ones((nn,))
        iv.add_output('q_in', val=10 * throttle_profile, units='kW')
        # iv.add_output('T_in', val=40*np.ones((nn_tot,)), units='degC')
        iv.add_output('mdot_coolant', val=0.1 * np.ones((nn,)), units='kg/s')
        iv.add_output('rho_coolant', val=997 * np.ones((nn,)), units='kg/m**3')
        iv.add_output('motor_mass', val=50., units='kg')
        iv.add_output('coolant_mass', val=10., units='kg')
        iv.add_output('T_motor_initial', val=15, units='degC')
        iv.add_output('T_res_initial', val=15.1, units='degC')
        iv.add_output('duration', val=800, units='s')
        iv.add_output('channel_width', val=1, units='mm')
        iv.add_output('channel_height', val=20, units='mm')
        iv.add_output('channel_length', val=0.2, units='m')
        iv.add_output('n_parallel', val=20)
        Ueas = np.ones((nn)) * 150
        h = np.concatenate([np.linspace(0, 25000, nn)])
        iv.add_output('fltcond|Ueas', val=Ueas, units='kn')
        iv.add_output('fltcond|h', val=h, units='ft')

        self.add_subsystem('atmos',
                           ComputeAtmosphericProperties(num_nodes=nn),
                           promotes_inputs=["fltcond|h",
                                            "fltcond|Ueas"])

        if not quasi_steady:
            lc_promotes = ['q_in', ('mass', 'motor_mass'), 'duration', 'channel_*', 'n_parallel']
        else:
            lc_promotes = ['q_in', 'channel_*', 'n_parallel']

        self.add_subsystem('component',
                           LiquidCooledComp(num_nodes=nn,
                                            quasi_steady=quasi_steady),
                           promotes_inputs=lc_promotes)
        self.add_subsystem('duct',
                           ImplicitCompressibleDuct(num_nodes=nn))

        self.connect('atmos.fltcond|p', 'duct.p_inf')
        self.connect('atmos.fltcond|T', 'duct.T_inf')
        self.connect('atmos.fltcond|Utrue', 'duct.Utrue')

        self.connect('component.T_out', 'duct.T_in_hot')
        self.connect('rho_coolant', 'duct.rho_hot')
        if quasi_steady:
            self.connect('duct.T_out_hot', 'component.T_in')
            self.connect('mdot_coolant', ['component.mdot_coolant', 'duct.mdot_hot'])
        else:
            self.add_subsystem('reservoir',
                               CoolantReservoir(num_nodes=nn),
                               promotes_inputs=['duration', ('mass', 'coolant_mass')])
            self.connect('duct.T_out_hot', 'reservoir.T_in')
            self.connect('reservoir.T_out', 'component.T_in')
            self.connect('mdot_coolant', ['component.mdot_coolant', 'duct.mdot_hot', 'reservoir.mdot_coolant'])
            self.connect('T_motor_initial', 'component.T_initial')
            self.connect('T_res_initial', 'reservoir.T_initial')