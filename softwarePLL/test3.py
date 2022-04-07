from threading import Thread, Event
import time
import matplotlib.pyplot as plt
import numpy as np


def get_signal_val(t_in_ns: float, frequenz_in_hz: float) -> float:
	"""Function that

		Args:
			t_in_ns: float,
			frequenz_in_hz:

		Returns:
			functional value of sinusoidal waveform
	"""
	return np.sin(2.0 * np.pi * frequenz_in_hz * 1E-9 * t_in_ns)


def generate_static_periodic_signal(t_in_ns: np.ndarray, frequenz_static_sig_in_hz: float) -> list:
	"""Function that

		Args:
			t_in_ns: float,
			frequenz_in_hz:

		Returns:
			entire sinusoidal waveform over input time array
	"""
	return np.sin(2.0 * np.pi * frequenz_static_sig_in_hz * 1E-9 * t_in_ns)


def coupling_function(t: float, own_state: float, ext_state: float, intrinsic_frequency: float):
	"""Function that

		Args:
			t_in_ns: float,
			frequenz_in_hz:

		Returns:
			entire sinusoidal waveform over input time array
	"""
	control_signal += 0.1 * intrinsic_frequency * (own_state * ext_state - control_signal) * (t - last_time)
	last_time = t
	return intrinsic_frequency + 0.1 * control_signal


# t = np.linspace(0, 2 * np.pi, 1000)


def get_phase(y: float, t_in_s: float, period_in_s: float) -> float:
	"""Function that returns the phase associated to the functional value of a sinusoidal signal.
		It uses the information about the origins' signal's current frequency to define a phase variable in [0, 2pi] from the arcsin.

		Args:
			y: functional value of a sinusoidal signal
			t_in_s: current machine time of the software PLL's signal in seconds
			period_in_s: the current period of the oscillation in seconds

		Returns:
			modified arcsin so that a phase in [0, 2pi) is achieved
		"""
	t_in_s %= period_in_s # obtain the fraction of the current period that has passed via modulo
	if t_in_s < period_in_s / 4:
		return np.arcsin(y)
	elif t_in_s >= period_in_s / 4 and t_in_s < period_in_s / 2:
		return -np.arcsin(y) + np.pi
	elif t_in_s >= period_in_s / 2 and t_in_s < 3 * period_in_s / 4:
		return -np.arcsin(y) + np.pi
	else:
		return np.arcsin(y) + 2 * np.pi


# phase = []
# for i in range(len(t)):
# 	phase.append(get_phase(np.sin(t[i]), t[i], 2 * np.pi))
# plot(t, phase, 'r')


class SharedData:
	run_osci: bool = True
	# frequenz_in_hz: np.ndarray = np.array([100, 100])
	# frequenz_in_hz: list = [1, 1]
	# print('{intrinsic frequency, period}={%0.1f, %0.10f}'%(frequenz_in_hz[0], 1/frequenz_in_hz[0]))
	intrinsic_freq_in_hz: float = 1
	frequenz_in_hz: float = 1
	print('{intrinsic frequency, period}={%0.1f, %0.10f}' % (intrinsic_freq_in_hz, 1 / intrinsic_freq_in_hz))


time_list: list = [0]
state_list: list = [0]
inst_ref_signal: list = [0]

counter: int = 0
max_update_counts: int = 3500000#0
update_in_ns: int = 25000  # 50000 -> 50 us

global periodic_ref_signal
tref = np.linspace(0, 15 * (max_update_counts - 1) * update_in_ns, 15 * max_update_counts)
fref = 1.1
taucc = 0.25E9
periodic_ref_signal = generate_static_periodic_signal(tref, fref)

#for reference signal: derive from the time directly to avoid problems with the counter!!!!

print('The total system time that this oscillation runs should be %0.5f s' % (max_update_counts * update_in_ns * 1E-9))

data: object = SharedData()
control_signal: float = 0
ctrlsig_list: list = [control_signal]
cut_off_frequency_hz: float = 0.01 * data.intrinsic_freq_in_hz
instantfreq_list: list = [data.frequenz_in_hz]

print('Fc = ', cut_off_frequency_hz, ' Hz, intrinsic frequency = ', data.intrinsic_freq_in_hz, ' SHz')

t: int = time.time_ns()
tinit: int = t

while data.run_osci:
	if time.time_ns() - t >= update_in_ns:
		t = time.time_ns()
		time_list.append(t - tinit)
		state_list.append(get_signal_val(t - tinit, data.frequenz_in_hz))
		inst_ref_signal.append(get_signal_val(t - tinit - taucc, fref))
		# = coupling_function(t-tinit, state_list[counter], periodic_ref_signal[counter], data.frequenz_in_hz)
		counter += 1
		# control_signal += cut_off_frequency_hz * (state_list[counter] * periodic_ref_signal[counter] - control_signal) * 1E-9 * (time_list[counter] - time_list[counter - 1])
		control_signal += cut_off_frequency_hz * (state_list[counter] * inst_ref_signal[counter] - control_signal) * 1E-9 * (time_list[counter] - time_list[counter - 1])
		# control_signal += cut_off_frequency_hz * (np.sin(get_phase(state_list[counter], time_list[counter], 1.0/data.frequenz_in_hz)-get_phase(periodic_ref_signal[counter], time_list[counter], 1.0/fref))-control_signal) * 1E-9 * (time_list[counter] - time_list[counter - 1])
		ctrlsig_list.append(control_signal)
		# print('control_signal=', control_signal)
		data.frequenz_in_hz = data.intrinsic_freq_in_hz + 0.2 * control_signal
		instantfreq_list.append(data.frequenz_in_hz)
	# print('data.frequenz_in_hz=', data.frequenz_in_hz)
	# time.sleep(1)
	if counter == max_update_counts:
		data.run_osci = False

time_diffs = [time_list[i + 1] - time_list[i] for i in range(len(time_list) - 1)]
phase_spll = [get_phase(state_list[i], time_list[i]*1E-9, 1.0/instantfreq_list[i]) for i in range(len(time_list) - 1)]
# phase_refo = [get_phase(periodic_ref_signal[i], time_list[i]*1E-9 ?? which, 1.0/instantfreq_list[i]) for i in range(len(time_list) - 1)]
phase_refo = [get_phase(inst_ref_signal[i], (time_list[i]-taucc)*1E-9, 1.0/fref) for i in range(len(time_list) - 1)]
print('Minimum time between two updates in {nanoseconds, microseconds, milliseconds}={%i, %0.6f, %0.9f}' % (np.min(time_diffs), np.min(time_diffs) * 1E-3, np.min(time_diffs) * 1E-6))
print('Maximum time between two updates in {nanoseconds, microseconds, milliseconds}={%i, %0.6f, %0.9f}' % (np.max(time_diffs), np.max(time_diffs) * 1E-3, np.max(time_diffs) * 1E-6))

plt.figure(1)
plt.plot(np.array(time_list) * 1E-9, state_list, 'b-+')
plt.xlabel('time in seconds')
plt.ylabel('s(t)')
plt.grid()

plt.figure(2)
#plt.plot(tref * 1E-9, periodic_ref_signal, 'g--')
plt.plot(np.array(time_list) * 1E-9, inst_ref_signal, 'g--')
plt.xlabel('time in seconds')
plt.ylabel('r(t)')
plt.grid()

plt.figure(3)
plt.plot(np.array(time_list) * 1E-9, instantfreq_list, 'b-')
plt.xlabel('time in seconds')
plt.ylabel('f(t)')
plt.grid()

plt.figure(4)
plt.plot(np.array(time_list) * 1E-9, ctrlsig_list, 'c-.')
plt.xlabel('time in seconds')
plt.ylabel('ctrl(t)')
plt.grid()

plt.figure(5)
plt.plot(np.array(time_list)[0:-1] * 1E-9, phase_spll, 'b-.', label='SPLL')
plt.plot(np.array(time_list)[0:-1] * 1E-9, phase_refo, 'r-.', label='REF')
plt.xlabel('time in seconds')
plt.ylabel('phases(t)')
plt.legend(loc='best')
plt.grid()

plt.draw()
plt.show()
