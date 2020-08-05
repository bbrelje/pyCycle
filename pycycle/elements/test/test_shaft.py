import numpy as np
import unittest
import os

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.elements.shaft import Shaft

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/shaft.csv",
                      delimiter=",", skiprows=1)

header = [
    'trqLoad1',
    'trqLoad2',
    'trqLoad3',
    'Nmech',
    'HPX',
    'fracLoss',
    'trqIn',
    'trqOut',
    'trqNet',
    'pwrIn',
    'pwrOut',
    'pwrNet']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))

from pycycle.elements.test.util import check_element_partials


class ShaftTestCase(unittest.TestCase):

    def setUp(self):
        self.top = Problem()
        self.top.model = Group()
        self.top.model.add_subsystem("shaft", Shaft(num_ports=3), promotes=["*"])

        self.top.model.set_input_defaults('trq_0', 17., units='ft*lbf')
        self.top.model.set_input_defaults('trq_1', 17., units='ft*lbf')
        self.top.model.set_input_defaults('trq_2', 17., units='ft*lbf')
        self.top.model.set_input_defaults('Nmech', 17., units='rpm')
        self.top.model.set_input_defaults('HPX', 17., units='hp')
        self.top.model.set_input_defaults('fracLoss', 17.)

        self.top.setup(check=False)

    def test_case1(self):
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            # input torques
            self.top['trq_0'] = data[h_map['trqLoad1']]
            self.top['trq_1'] = data[h_map['trqLoad2']]
            self.top['trq_2'] = data[h_map['trqLoad3']]

            # shaft inputs
            self.top['Nmech'] = data[h_map['Nmech']]
            self.top['HPX'] = data[h_map['HPX']]
            self.top['fracLoss'] = data[h_map['fracLoss']]
            self.top.run_model()

            # check outputs
            trqIn, trqOut, trqNet = data[
                h_map['trqIn']], data[
                h_map['trqOut']], data[
                h_map['trqNet']]
            pwrIn, pwrOut, pwrNet = data[
                h_map['pwrIn']], data[
                h_map['pwrOut']], data[
                h_map['pwrNet']]
            trqIn_comp = self.top['trq_in']
            trqOut_comp = self.top['trq_out']
            trqNet_comp = self.top['trq_net']
            pwrIn_comp = self.top['pwr_in']
            pwrOut_comp = self.top['pwr_out']
            pwrNet_comp = self.top['pwr_net']

            tol = 1.0e-4
            assert_near_equal(trqIn_comp, trqIn, tol)
            assert_near_equal(trqOut_comp, trqOut, tol)
            assert_near_equal(trqNet_comp, trqNet, tol)
            assert_near_equal(pwrIn_comp, pwrIn, tol)
            assert_near_equal(pwrOut_comp, pwrOut, tol)
            assert_near_equal(pwrNet_comp, pwrNet, tol)

            check_element_partials(self, self.top)

if __name__ == "__main__":
    unittest.main()
