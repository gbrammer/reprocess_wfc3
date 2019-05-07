import unittest

import numpy as np
from .. import utils

class Dummy(unittest.TestCase):  
    def test_nmad(self):
        np.random.seed(1)
        N = 10000
        rnd = np.random.normal(size=N)
        np.testing.assert_allclose(utils.nmad(rnd), 1., rtol=3*np.sqrt(N), atol=0, equal_nan=False, err_msg='', verbose=True)
