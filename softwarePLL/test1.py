from threading import Thread, Event
import time
import matplotlib.pyplot as plt
import numpy as np


def get_signal_val(t, frequenz_in_hz):
	return np.sin(2.0*np.pi*frequenz_in_hz*1E-9*t)


def run_oscillations(data):
	counter: int = 0
	max_update_counts: int = 100000
	update_in_ns: int = 10

	print('The total system time that this oscillation runs should be %0.5f s' % (max_update_counts * update_in_ns * 1E-9))

	t = time.time_ns()
	tinit = t

	while data.run_osci:
		if time.time_ns() - t >= update_in_ns:
			t = time.time_ns() - tinit
			time_list.append(t)
			# print('counter', max_update_counts-counter)
			state_list.append(get_signal_val(t, data.frequenz_in_hz))
			counter += 1
		if counter == max_update_counts:
			data.run_osci = False


class SharedData:
	run_osci: bool = True
	frequenz_in_hz: float = 10000
	print('{intrinsic frequency, period}={%0.1f, %0.10f}'%(frequenz_in_hz, 1/frequenz_in_hz))


time_list = []
state_list = []

update_in_ns: int = 100

data = SharedData()
t = time.time_ns()
tinit = t
state = get_signal_val(t-tinit, data.frequenz_in_hz)

thread = Thread(target=run_oscillations, args=(data, ))
thread.start()

t = time.time_ns()
while data.run_osci:
	if time.time_ns() - t >= update_in_ns*100:
		data.frequenz_in_hz = data.frequenz_in_hz

thread.join()

plt.plot(np.array(time_list)*1E-9, state_list, 'b-*')
plt.xlabel('time in seconds')
plt.ylabel('s(t)')
plt.grid()
plt.draw()
plt.show()
