"""
Test for the SetStaticArea component.
"""
import os.path
import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.species_data import janaf, Thermo
from pycycle.cea.set_total import SetTotal
from pycycle.cea.set_static import SetStatic
from pycycle import constants

fpath = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(fpath, 'NPSS_Static_CEA_Data.csv')
ref_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

header = ['W', 'MN', 'V', 'A', 's', 'Pt', 'Tt', 'ht', 'rhot',
          'gamt', 'Ps', 'Ts', 'hs', 'rhos', 'gams']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class TestSetStaticArea(unittest.TestCase):

    def test_case_Area(self):

        thermo = Thermo(janaf, init_reacts=constants.AIR_MIX)

        p = Problem()

        p.model.add_subsystem('set_total_TP', SetTotal(thermo_data=janaf), promotes_inputs=['b0', 'P'])
        p.model.add_subsystem('set_static_A', SetStatic(mode='area', thermo_data=janaf), promotes_inputs=['b0', ('guess:Pt', 'P')])
        p.model.set_input_defaults('b0', thermo.b0)
        p.model.set_input_defaults('set_total_TP.T', val=518., units='degR')

        p.model.set_input_defaults('P', val=14.7, units='psi')
        p.model.set_input_defaults('set_static_A.W', val=1.5, units='lbm/s')
        p.model.set_input_defaults('set_static_A.area', val=np.inf, units='inch**2')

        p.model.connect('set_total_TP.flow:S', 'set_static_A.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_A.ht')
        p.model.connect('set_total_TP.flow:gamma', 'set_static_A.guess:gamt')

        p.set_solver_print(level=-1)
        p.setup(check=False)

        # 4 cases to check against
        for i, data in enumerate(ref_data):

            p['set_total_TP.T'] = data[h_map['Tt']]
            p['P'] = data[h_map['Pt']]

            p['set_static_A.area'] = data[h_map['A']]
            p['set_static_A.W'] = data[h_map['W']]

            if i is 5:  # supersonic case
                p['set_static_A.guess:MN'] = 3.

            p.run_model()

            # check outputs
            npss_vars = ('Ps', 'Ts', 'MN', 'hs', 'rhos', 'gams', 'V', 'A', 's', 'ht')
            Ps, Ts, MN, hs, rhos, gams, V, A, S, ht = tuple(
                [data[h_map[v_name]] for v_name in npss_vars])

            Ps_computed = p['set_static_A.flow:P']
            Ts_computed = p['set_static_A.flow:T']
            hs_computed = p['set_static_A.flow:h']
            rhos_computed = p['set_static_A.flow:rho']
            gams_computed = p['set_static_A.flow:gamma']
            V_computed = p['set_static_A.flow:V']
            A_computed = p['set_static_A.area']
            MN_computed = p['set_static_A.MN']

            tol = 1.0e-4
            assert_near_equal(gams_computed, gams, tol)
            assert_near_equal(MN_computed, MN, tol)
            assert_near_equal(Ps_computed, Ps, tol)
            assert_near_equal(Ts_computed, Ts, tol)
            assert_near_equal(hs_computed, hs, tol)
            assert_near_equal(rhos_computed, rhos, tol)
            assert_near_equal(gams_computed, gams, tol)
            assert_near_equal(V_computed, V, tol)
            assert_near_equal(A_computed, A, tol)


if __name__ == "__main__":
    import scipy

    np.seterr(all='raise')
    unittest.main()
