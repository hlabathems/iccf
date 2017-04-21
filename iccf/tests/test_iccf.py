# Licensed under a 3-clause BSD style license - see Licence.rst
# This module implements tests for ICCF

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from numpy.testing import assert_array_equal

from ..iccf import *

def test_to_flux_err():
    err_mag = 0.1
    counts = 5e-14
    assert to_flux_err(err_mag, counts)==(err_mag * np.log(10) * counts) / 2.5

def test_to_flux_err_array():
    err_mag = np.array([0.1, 0.1])
    counts = np.array([5e-14, 6e-14])
    flux_err = err_mag * np.log(10) * counts / 2.5
    assert_array_equal(to_flux_err(err_mag, counts), flux_err)
