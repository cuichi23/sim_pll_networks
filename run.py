#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import dicts_lib
import sim_lib
import setup

# obtain dictionaries for the PLLs and the network
dictPLL, dictNet            = dicts_lib.getDicts()
# print('dictPLL:', dictPLL)
# print('dictNet:', dictNet)

# simulate the network of coupled PLLs
dictNet, dictPLL, dictData  = sim_lib.simulateSystem(dictNet, dictPLL)
