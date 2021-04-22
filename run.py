#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import dicts_lib, dicts_lib_Ising, dicts_lib_3rdGen, dicts_lib_4thGen, dicts_lib_Nav
import synctools_interface_lib as synctools
import sim_lib
import setup

# obtain dictionaries for the PLLs and the network
#dictPLL, dictNet            = dicts_lib_4thGen.getDicts()
#dictPLL, dictNet            = dicts_lib_3rdGen.getDicts()
#dictPLL, dictNet            = dicts_lib_Ising.getDicts()
#dictPLL, dictNet            = dicts_lib_Nav.getDicts()
dictPLL, dictNet            = dicts_lib.getDicts()
# print('dictPLL:', dictPLL)
# print('dictNet:', dictNet)


# simulate the network of coupled PLLs
dictNet, dictPLL, dictData  = sim_lib.simulateSystem(dictNet, dictPLL)
