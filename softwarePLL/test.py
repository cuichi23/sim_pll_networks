from threading import Thread, Event
import time
import matplotlib.pyplot as plt
import numpy as np


def modify_variable(data: object, state: float, update_in_ns: int):
	#t: int = time.time_ns()
	#counter: int = 0
	while data.run_osci:
		if time.time_ns() - t >= update_in_ns*1000:
			#counter += 1
			#data.frequenz_in_hz[(data.counter+1)%2] -= 0.0001
			data.frequenz_in_hz -= 0.0001

def get_signal_val(t: float, frequenz_in_hz: float):
	return np.sin(2.0*np.pi*frequenz_in_hz*1E-9*t)


class SharedData:
	run_osci: bool = True
	# frequenz_in_hz: np.ndarray = np.array([100, 100])
	# frequenz_in_hz: list = [1000, 1000]
	frequenz_in_hz: float = 1000
	counter = 0
	# print('{intrinsic frequency, period}={%0.1f, %0.10f}'%(frequenz_in_hz[0], 1/frequenz_in_hz[0]))
	print('{intrinsic frequency, period}={%0.1f, %0.10f}' % (frequenz_in_hz, 1 / frequenz_in_hz))


time_list: list  = []
state_list: list = []

#counter: int = 0
max_update_counts: int = 1000000
update_in_ns: int = 50000 # 50000 -> 50 us

print('The total system time that this oscillation runs should be %0.5f s'%(max_update_counts*update_in_ns*1E-9))

data: object = SharedData()
t: int = time.time_ns()
tinit: int = t
# state: float = get_signal_val(t-tinit, data.frequenz_in_hz[0])
state: float = get_signal_val(t-tinit, data.frequenz_in_hz)


thread = Thread(target=modify_variable, args=(data, state, update_in_ns))
thread.start()

while data.run_osci:
	if time.time_ns() - t >= update_in_ns:
		t = time.time_ns() - tinit
		time_list.append(t)
		# print('counter', max_update_counts-counter)
		#state_list.append(get_signal_val(t, data.frequenz_in_hz[data.counter%2]))
		state_list.append(get_signal_val(t, data.frequenz_in_hz))
		data.counter += 1
	if data.counter == max_update_counts:
		data.run_osci = False

thread.join()


time_diffs = [time_list[i+1]-time_list[i] for i in range(len(time_list)-1)]
print('Minimum time between two updates in {nanoseconds, microseconds, milliseconds}={%i, %0.6f, %0.9f}' % (np.min(time_diffs), np.min(time_diffs)*1E-3, np.min(time_diffs)*1E-6))
print('Maximum time between two updates in {nanoseconds, microseconds, milliseconds}={%i, %0.6f, %0.9f}' % (np.max(time_diffs), np.max(time_diffs)*1E-3, np.max(time_diffs)*1E-6))

plt.plot(np.array(time_list)*1E-9, state_list, 'b-*')
plt.xlabel('time in seconds')
plt.ylabel('s(t)')
plt.grid()
plt.draw()
plt.show()
