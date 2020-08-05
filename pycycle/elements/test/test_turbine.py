""" Tests the turbine component. """

import os
import unittest

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.species_data import janaf, Thermo
from pycycle.connect_flow import connect_flow
from pycycle.constants import AIR_MIX
from pycycle.elements.turbine import Turbine
from pycycle.elements.flow_start import FlowStart
from pycycle.maps.lpt2269 import LPT2269

from pycycle.elements.test.util import check_element_partials

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/turbine.csv", delimiter=",", skiprows=1)

header = [
    'Fl_I.W',
    'Fl_I.Pt',
    'Fl_I.Tt',
    'Fl_I.ht',
    'Fl_I.s',
    'Fl_I.MN',
    'Fl_O.Pt',
    'Fl_O.Tt',
    'Fl_O.ht',
    'Fl_O.s',
    'Fl_O.MN',
    'PRdes',
    'EffDes',
    'Power',
    'Nmech',
    'Fl_O.Ps',
    'Fl_O.Ts',
    'Fl_O.hs',
    'Fl_O.rhos',
    'Fl_O.gams',
    'effPoly']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class TurbineTestCase(unittest.TestCase):

    def setUp(self):

        thermo = Thermo(janaf, AIR_MIX)
        self.prob = Problem()
        self.prob.model = Group()

        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.add_subsystem('turbine', Turbine(map_data=LPT2269, design=True,
                                                         elements=AIR_MIX))

        self.prob.model.set_input_defaults('turbine.Nmech', 1000.0, units='rpm')
        self.prob.model.set_input_defaults('flow_start.MN', 0.0)
        self.prob.model.set_input_defaults('turbine.PR', 4.0)
        self.prob.model.set_input_defaults('turbine.MN', 0.0)
        self.prob.model.set_input_defaults('flow_start.P', 17.0, units='psi')
        self.prob.model.set_input_defaults('flow_start.T', 500.0, units='degR')
        self.prob.model.set_input_defaults('flow_start.W', 100.0, units='lbm/s')
        self.prob.model.set_input_defaults('turbine.eff', 0.9)

        connect_flow(self.prob.model, 'flow_start.Fl_O', 'turbine.Fl_I')

        self.prob.setup(check=False)
        self.prob.set_solver_print(level=-1)

    def test_case1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):

            print()
            print('---- Test Case,', i, '----')

            self.prob['turbine.PR'] = data[h_map['PRdes']]
            self.prob['turbine.Nmech'] = data[h_map['Nmech']]
            self.prob['turbine.eff'] = data[h_map['EffDes']]
            self.prob['turbine.MN'] = data[h_map['Fl_O.MN']]

            # input flowstation
            self.prob['flow_start.P'] = data[h_map['Fl_I.Pt']]
            self.prob['flow_start.T'] = data[h_map['Fl_I.Tt']]
            self.prob['flow_start.W'] = data[h_map['Fl_I.W']]
            self.prob['flow_start.MN'] = data[h_map['Fl_I.MN']]

            self.prob.run_model()

            # check outputs
            tol = 5e-4

            npss = data[h_map['EffDes']]
            pyc  = self.prob['turbine.eff'][0]
            print('EffDes:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_I.s']]
            pyc  = self.prob['flow_start.Fl_O:tot:S'][0]
            print('Fl_I.s:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_I.Tt']]
            pyc  = self.prob['flow_start.Fl_O:tot:T'][0]
            print('Fl_I.Tt:', npss, pyc)
            assert_near_equal(npss, pyc , tol)

            npss = data[h_map['Fl_I.ht']]
            pyc  = self.prob['flow_start.Fl_O:tot:h'][0]
            print('Fl_I.ht:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_I.W']]
            pyc  = self.prob['flow_start.Fl_O:stat:W'][0]
            print('Fl_I.W:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.MN']]
            pyc  = self.prob['turbine.MN'][0]
            print('Fl_O.MN:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.Pt']]
            pyc  = self.prob['turbine.Fl_O:tot:P'][0]
            print('Fl_O.Pt:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.ht']]
            pyc  = self.prob['turbine.Fl_O:tot:h'][0]
            print('Fl_O.ht:', npss, pyc)
            assert_near_equal(npss, pyc, tol)

            npss = data[h_map['Fl_O.Tt']]
            pyc  = self.prob['turbine.Fl_O:tot:T'][0]
            print('Fl_O.Tt:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.ht']] - data[h_map['Fl_I.ht']]
            pyc  = self.prob['turbine.Fl_O:tot:h'][0] -self.prob['flow_start.Fl_O:tot:h'][0]
            print('Fl_O.ht - Fl_I.ht:', npss, pyc)
            assert_near_equal(npss , pyc ,tol)

            npss = data[h_map['Fl_O.s']]
            pyc  = self.prob['turbine.Fl_O:tot:S'][0]
            print('Fl_O.s:', npss, pyc)
            assert_near_equal(npss, pyc, tol)

            npss = data[h_map['Power']]
            pyc  = self.prob['turbine.power'][0]
            print('Power:', npss, pyc)
            assert_near_equal(npss, pyc , tol)

            npss = data[h_map['Fl_O.Ps']]
            pyc  = self.prob['turbine.Fl_O:stat:P'][0]
            print('Fl_O.Ps:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.Ts']]
            pyc  = self.prob['turbine.Fl_O:stat:T'][0]
            print('Fl_O.Ts:', npss, pyc)
            assert_near_equal(npss, pyc, tol)

            npss = data[h_map['effPoly']]
            pyc  = self.prob['turbine.eff_poly'][0]
            print('effPoly:', npss, pyc)
            assert_near_equal(npss, pyc, tol)

            check_element_partials(self, self.prob, tol=1e-4)

if __name__ == "__main__":
    np.seterr(divide='warn')
    unittest.main()
