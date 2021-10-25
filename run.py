#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import dicts_lib, dicts_lib_Ising, dicts_lib_3rdGen, dicts_lib_4thGen, dicts_lib_test, dicts_lib_Nav, dicts_lib_FPEexplore
import synctools_interface_lib as synctools
import multisim_lib
import sim_lib
import setup

# multiprocessing based simulations
#multiprocess = True
multiprocess = False

# obtain dictionaries for the PLLs and the network
#dictPLL, dictNet, dictAlgo            = dicts_lib_FPEexplore.getDicts()
#dictPLL, dictNet, dictAlgo            = dicts_lib_4thGen.getDicts()
#dictPLL, dictNet, dictAlgo            = dicts_lib_3rdGen.getDicts()
#dictPLL, dictNet, dictAlgo            = dicts_lib_Ising.getDicts()
#dictPLL, dictNet, dictAlgo            = dicts_lib_test.getDicts()
#dictPLL, dictNet, dictAlgo            = dicts_lib_Nav.getDicts()
dictPLL, dictNet, dictAlgo            = dicts_lib.getDicts()
# print('dictPLL:', dictPLL)
# print('dictNet:', dictNet)
# print('dictNet:', dictAlgo)

# simulate the network of coupled PLLs
if multiprocess:
	poolData  					= multisim_lib.distributeProcesses(dictNet, dictPLL, dictAlgo)
elif not multiprocess:
	dictNet, dictPLL, dictData	= sim_lib.simulateSystem(dictNet, dictPLL)
