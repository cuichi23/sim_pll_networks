from threading import Thread, Event
import time
import matplotlib.pyplot as plt
import numpy as np


def get_signal_val(t: float, frequenz_in_hz: float):
	return np.sin(2.0*np.pi*frequenz_in_hz*1E-9*t)


def coupling_function(own_state: float, instantaneous_frequency: float):
	return instantaneous_frequency + 0.0


class SharedData:
	run_osci: bool = True
	# frequenz_in_hz: np.ndarray = np.array([100, 100])
	# frequenz_in_hz: list = [1, 1]
	# print('{intrinsic frequency, period}={%0.1f, %0.10f}'%(frequenz_in_hz[0], 1/frequenz_in_hz[0]))
	frequenz_in_hz: float = 1
	print('{intrinsic frequency, period}={%0.1f, %0.10f}' % (frequenz_in_hz, 1 / frequenz_in_hz))


time_list: list  = []
state_list: list = []

counter: int = 0
max_update_counts: int = 1000000
update_in_ns: int = 2500 # 50000 -> 50 us

print('The total system time that this oscillation runs should be %0.5f s'%(max_update_counts*update_in_ns*1E-9))

data: object = SharedData()
t: int = time.time_ns()
tinit: int = t

while data.run_osci:
	if time.time_ns() - t >= update_in_ns:
		t = time.time_ns()
		time_list.append(t-tinit)
		state_list.append(get_signal_val(t-tinit, data.frequenz_in_hz))
		data.frequenz_in_hz = coupling_function(state_list[counter], data.frequenz_in_hz)
		counter += 1
	if counter == max_update_counts:
		data.run_osci = False

time_diffs = [time_list[i+1]-time_list[i] for i in range(len(time_list)-1)]
print('Minimum time between two updates in {nanoseconds, microseconds, milliseconds}={%i, %0.6f, %0.9f}' % (np.min(time_diffs), np.min(time_diffs)*1E-3, np.min(time_diffs)*1E-6))
print('Maximum time between two updates in {nanoseconds, microseconds, milliseconds}={%i, %0.6f, %0.9f}' % (np.max(time_diffs), np.max(time_diffs)*1E-3, np.max(time_diffs)*1E-6))

plt.plot(np.array(time_list)*1E-9, state_list, 'b-*')
plt.xlabel('time in seconds')
plt.ylabel('s(t)')
plt.grid()
plt.draw()
plt.show()
