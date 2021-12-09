import numpy as np
np.random.seed(19680801)
import matplotlib.pyplot as plt
cm = plt.cm.get_cmap('RdYlBu')
xy = range(20)
z = xy
sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)
plt.colorbar(sc)






fig1, ax1 = plt.subplots()
for color in ['tab:blue', 'tab:orange', 'tab:green']:
    n = 750
    x, y = np.random.rand(2, n)
    scale = 200.0 * np.random.rand(n)
    ax1.scatter(x, y, c=color, s=scale, label=color,
               alpha=0.3, edgecolors='none')

ax1.legend()
ax1.grid(True)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Generate torus mesh
angle = np.linspace(0, 2 * np.pi, 32)
theta, phi = np.meshgrid(angle, angle)
r, R = .25, 1.
X = (R + r * np.cos(phi)) * np.cos(theta)
Y = (R + r * np.cos(phi)) * np.sin(theta)
Z = r * np.sin(phi)

# Display the mesh
fig2 = plt.figure()
ax2 = fig2.gca(projection = '3d')
ax2.set_xlim3d(-1, 1)
ax2.set_ylim3d(-1, 1)
ax2.set_zlim3d(-1, 1)
ax2.plot_surface(X, Y, Z, color = 'w', rstride = 1, cstride = 1)
plt.show()






























def loop_K_fR_TR(facTtotal, TR_vec, refsig_edgetimes, mulow, muhigh, VCOinit, tau, tau_f, fR_vec, y0, K_vec, l, alpha, t, dutyRef, results):
	for index1 in range(len(fR_vec)):
		refsig_edgetimes = setup_refsig(alpha, t, fR_vec[index1], dutyRef)['ref_edgetimes'];
		for index2 in range(len(K_vec)):
			# prepare simulation
			tempVCO			 = start_VCO(VCOinit, mulow, muhigh);
			vcoStateHigh	 = tempVCO['vcoState'];
			Tvco_edgepoints	 = tempVCO['Tvco_egdes'];
			Tvco_edge		 = tempVCO['first_vco_edge'];
			Tvco			 = tempVCO['Tvco_edgetimes'];
			# simulate realization
			sim_results = simRef2PLL_discrete(facTtotal*TR_vec[index1], refsig_edgetimes, Tvco, Tvco_edge, Tvco_edgepoints, vcoStateHigh, mulow, muhigh, tau, tau_f, fR_vec[index1], y0, K_vec[index2], l);
			Tvco	= sim_results['VCO_edge_times'];
			Tpd 	= sim_results['PD_edge_times'];
			solCtrl = sim_results['control_sig'];
			# calculate phases and derivative of phases from sets of edge-times
			phi_tvco = edgtimes2phase(Tvco, 0.0);
			phi_ref  = edgtimes2phase(refsig_edgetimes, alpha);
			# interpolate to get common phase-time series to calculate the Kuramoto order parameter
			# I use linear interpolation, since I assume linear growth of the phase between two edges,
			# also according to the model in which the frequency only changes at each output edge event
			phi_tvco_interpol = interp1d(phi_tvco['time'], phi_tvco['phi'], kind='linear') # this defines the basis using the available
			phi_ref_interpol  = interp1d(phi_ref['time'],  phi_ref['phi'] )				   # discrete phase evolution points
			#tvco=np.linspace(phi_tvco['time'][0], phi_tvco['time'][-1], interpolation_res) # create time-vector for the length of the
			#tref=np.linspace(phi_ref['time'][0], phi_ref['time'][-1], interpolation_res)   # available data using start and end time
			#print('tvco[0], tref[0]', tvco[0], tref[0], 'tvco[-1], tref[-1]', tvco[-1], tref[-1])
			if phi_tvco['time'][0] != phi_ref['time'][0]:
				print('CAREFUL, check what to do with interpolation, starting times are different!')
			min_time_inter=max(phi_tvco['time'][0] , phi_ref['time'][0] );				# find the min-time for the common vector to be formed
			max_time_inter=min(phi_tvco['time'][-1], phi_ref['time'][-1]);				# find the max-time of both for the common vector
			tinter = np.linspace(min_time_inter, max_time_inter, interpolation_res)		# the time-vector for the interpolation of all phi-vectors
			phi_vco_inter = np.array(phi_tvco_interpol(tinter));						#print('phi_vco_interol:', phi_vco_interol.ndim, phi_vco_interol.shape, '\n', phi_vco_interol)
			phi_ref_inter = np.array(phi_ref_interpol(tinter));							#print('phi_ref_interol:', phi_ref_interol.ndim, phi_ref_interol.shape, '\n', phi_ref_interol)
			phases   	  = np.vstack( ( phi_vco_inter, phi_ref_inter) );				#phases   = np.vstack( ( np.array(phi_tvco['phi'].copy()), np.array(phi_ref['phi'][:-1].copy()) ) ).T;
			phase_diff	  = np.fmod(phi_vco_inter-phi_ref_inter+np.pi, 2.0*np.pi)-np.pi;
			#print('phases:', phases.ndim, phases.shape, '\n', phases)
			orderP   = kuramotoOrderP(phases)
			# the goal is to average certain measurements over a time given by 'averagOver = X * TR', where X is integer>0
			s_average_index = np.argmax( tinter >= (tinter[-1]-((averagOver/TR) * TR_vec[index1])) ); # averagOver/TR_vec just yields the factor specified in the header
			#print('test: s-averaging over a time of ', tinter[-1]-tinter[s_average_index],' seconds, and',(averagOver/TR),'* TR, while TR[n] =', TR_vec[index1])
			#print('that results in s_average_index:', s_average_index, ' at a total length of tinter of', len(tinter))
			averaging_time	= tinter[-1]-tinter[s_average_index];
			vco_inst_freq   = edgetimes2freq(Tvco); freq_diff = vco_inst_freq['freqFP'][:] - fR_vec[index1];
			f_average_index = np.argmax( vco_inst_freq['timeFP'] >= (vco_inst_freq['timeFP'][-1] - ((averagOver/TR) * TR_vec[index1])) );
			#print('test: f-averaging over a time of ', vco_inst_freq['time'][-1]-vco_inst_freq['time'][f_average_index],' seconds, and',(averagOver/TR),'* TR, while TR[n] =', TR_vec[index1])
			#print(fR_vec[index2], K_vec[index2], orderP['order'][-1], np.mean(orderP['order'][-s_average_index:]), np.var(orderP['order'][-s_average_index:]))
			# determine whether the state is frequency-locked or not, i.e., whether the phases are locked to each other
			# stratgy 1: of the order parameter is constant, then the phase-relation must be fixed (small systems)
			boolean_stability = check_solution(orderP, fR_vec[index1], vco_inst_freq, averaging_time, f_average_index, s_average_index);

			results.append([TR_vec[index1], fR_vec[index1], K_vec[index2], orderP['order'][-1],
							np.mean(orderP['order'][s_average_index:]), np.var(orderP['order'][s_average_index:]),
							phase_diff[-1], np.mean(phase_diff[s_average_index:]), np.var(phase_diff[s_average_index:]),
							boolean_stability ,averaging_time, freq_diff[-1], np.mean(freq_diff[-f_average_index:]),
							np.var(freq_diff[-f_average_index:])]);
	return {'results': np.array(results), 'tinter': tinter}
