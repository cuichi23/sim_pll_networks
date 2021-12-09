import networkx as nx
import sys
# from numpy import linalg as LA
from scipy import linalg as la
from numpy import pi, sin
import numpy as np
import math
import sympy
from sympy import solve, nroots, I
from sympy.abc import q
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time

def eigenvalzeta(topology, Nx=3, Ny=3):
	global G;
	Nplls=Ny*Nx
	if topology == 'ring':
		G = nx.cycle_graph(Nplls)

	elif topology == 'chain':
		G = nx.path_graph(Nplls)
	else:
		N = np.sqrt(Nplls)
		if Nx == Ny:
			if N.is_integer():             # indirect check, whether N is an integer, which it should be for Nx=Ny as above checked
				N = int(N)
			else:
				raise ValueError('Npll is not valid: sqrt(N) is not an integer')

	if topology == 'square-open':

		Ny = int(Ny)
		Nx = int(Nx)
		G  = nx.grid_2d_graph(Nx,Ny)

	elif topology == 'square-periodic':
		Ny = np.sqrt(Nplls)
		Nx = np.sqrt(Nplls)
		Ny = int(Ny)
		Nx = int(Nx)
		G  = nx.grid_2d_graph(Nx,Ny, periodic=True)                            # for periodic boundary conditions:

	elif topology == 'hexagon':
		print('\nIf Nx =! Ny, then check the graph that is generated again!')
		G=nx.grid_2d_graph(Nx,Ny)           # why not ..._graph(Nx,Ny) ? NOTE the for n in G: loop has to be changed, loop over nx and ny respectively, etc....
		for n in G:
			x,y=n
			if x>0 and y>0:
				G.add_edge(n,(x-1,y-1))
			if x<Nx-1 and y<Ny-1:
				G.add_edge(n,(x+1,y+1))

	elif topology == 'hexagon-periodic':
		G=nx.grid_2d_graph(Nx,Ny, periodic=True)
		for n in G:
			x,y=n
			G.add_edge(n, ((x-1)%Nx, (y-1)%Ny))

	elif topology == 'octagon':            # why not ..._graph(Nx,Ny) ? NOTE the for n in G: loop has to be changed, loop over nx and ny respectively, etc....
		print('\nIf Nx =! Ny, then check the graph that is generated again!')
		G=nx.grid_2d_graph(Nx,Ny)
		for n in G:
			x,y=n
			if x>0 and y>0:
				G.add_edge(n,(x-1,y-1))
			if x<Nx-1 and y<Ny-1:
				G.add_edge(n,(x+1,y+1))
			if x<Nx-1 and y>0:
				G.add_edge(n,(x+1,y-1))
			if x<Nx-1 and y>0:
				G.add_edge(n,(x+1,y-1))
			if x>0 and y<Ny-1:
				G.add_edge(n,(x-1,y+1))

	elif topology == 'octagon-periodic':
		G=nx.grid_2d_graph(Nx,Ny, periodic=True)
		for n in G:
			x,y=n
			G.add_edge(n, ((x-1)%Nx, (y-1)%Ny))
			G.add_edge(n, ((x-1)%Nx, (y+1)%Ny))
			# G = nx.convert_node_labels_to_integers(G)

	#G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted') # converts 2d coordinates to 1d index of integers, e.g., k=0,...,N-1
	if not (topology=='global' or topology=='global-with-selfcoup'):
		if topology == 'ring':
			A=(1.0/2.0)*nx.to_numpy_matrix(G)
		elif topology == 'square-periodic':
			A=(1.0/4.0)*nx.to_numpy_matrix(G);
		elif topology == 'hexagon-periodic':
			A=(1.0/6.0)*nx.to_numpy_matrix(G)
		elif topology == 'octagon-periodic':
			A=(1.0/8.0)*nx.to_numpy_matrix(G)
		elif topology == 'chain':
			#G.edges()
			#nx.draw(G); plt.show();
			A=nx.to_numpy_matrix(G)
			for i in range(Nplls):
				for j in range(Nplls):
					if not ((i==0 and j==1) or (i==Nplls-1 and j==Nplls-2)):
						A[i,j]=A[i,j]/2.0;

			print('chain:\n', A)

		elif topology == 'square-open': #here we construct the coupling matrix for open 2d Square case. We cannot change the weight
			A=nx.to_numpy_matrix(G)
			print('before:\n', A, '\n\n')
			nx.draw(G); plt.show();
			for i in range(Nplls):
				for j in range(Nplls):
					if (j==0 or j==Nplls-1 or j==Nx-1 or j==Nplls-Nx):
						A[i,j]=A[i,j]/2.0;
					elif ((j>0 and j<Nx-1) or (j>Nplls-Nx and j<Nplls-1) or np.mod(j,Nx)==0 or np.mod(j-(Nx-1),Nx)==0):
						A[i,j]=A[i,j]/3.0;

			print('square-open:\n', A)

			for i in range(Nx):
				for j in range(Ny):
					if (i==0 and j==0): #or (i==Nplls and j==Nplls) or (i==0 and j==Nplls) or (i==Nplls and j==0)):
						# G.has_edge(0, 1)
						G.remove_edge((i,j),(i,j+1))
						G.remove_edge((i,j),(i+1,j))
						# A=nx.to_numpy_matrix(G)
						# print(A)
						G.add_edge((i,j),(i,j+1), weight=1./2.)
						G.add_edge((i,j),(i+1,j), weight=1./2.)
						# A=nx.to_numpy_matrix(G)
						# print(A)
					elif (i==Nx-1 and j==Ny-1):# or (i==0 and j==Nplls) or (i==Nplls and j==0)):
						# G.has_edge(0, 1)
						G.remove_edge((i,j),(i,j-1))
						G.remove_edge((i,j),(i-1,j))
						G.add_edge((i,j),(i,j-1), weight=1./2.)
						G.add_edge((i,j),(i-1,j), weight=1./2.)
					elif (i==0 and j==Ny-1):# or (i==0 and j==Nplls) or (i==Nplls and j==0)):
						# G.has_edge(0, 1)
						G.remove_edge((i,j),(i,j-1))
						G.remove_edge((i,j),(i+1,j))
						# A=nx.to_numpy_matrix(G)
						# print(A)
						G.add_edge((i,j),(i,j-1), weight=1./2.)
						G.add_edge((i,j),(i+1,j), weight=1./2.)
					elif (i==Nx-1 and j==0):# or (i==0 and j==Nplls) or (i==Nplls and j==0)):
							# G.has_edge(0, 1)
						G.remove_edge((i,j),(i,j+1))
						G.remove_edge((i,j),(i-1,j))
						G.add_edge((i,j),(i,j+1), weight=1./2.)
						G.add_edge((i,j),(i-1,j), weight=1./2.)
						# A=nx.to_numpy_matrix(G)
						# print(A)

					elif ((i!=0 and j==0) or (i!=Nx-1 and j==0) or (i!=0 and j==Ny-1) or (i!=Nx-1 and j==Ny-1)): #or (i==Nplls and j==Nplls) or (i==0 and j==Nplls) or (i==Nplls and j==0)):
						# G.has_edge(0, 1)
						G.remove_edge((i,j),(i-1,j))
						G.remove_edge((i,j),(i+1,j))
						G.add_edge((i,j),(i-1,j), weight=1./3.)
						G.add_edge((i,j),(i+1,j), weight=1./3.)
						# A=nx.to_numpy_matrix(G)
						# print(A)
					elif((i==0 and j!=0) or (i==0 and j!=Ny-1) or (i==Nx-1 and j==0) or (i==Nx-1 and j!=Ny-1)): #or (i==Nplls and j==Nplls) or (i==0 and j==Nplls) or (i==Nplls and j==0)):
						# G.has_edge(0, 1)
						G.remove_edge((i,j),(i,j-1))
						G.remove_edge((i,j),(i,j+1))
						G.add_edge((i,j),(i,j-1), weight=1./3.)
						G.add_edge((i,j),(i,j+1), weight=1./3.)
						# A=nx.to_numpy_matrix(G)
						# print(A)
					else:
						G.remove_edge((i,j),(i,j-1))
						G.remove_edge((i,j),(i,j+1))
						G.remove_edge((i,j),(i-1,j))
						G.remove_edge((i,j),(i+1,j))
						G.add_edge((i,j),(i,j-1), weight=1./4.)
						G.add_edge((i,j),(i,j+1), weight=1./4.)
						G.add_edge((i,j),(i-1,j), weight=1./4.)
						G.add_edge((i,j),(i+1,j), weight=1./4.)

				# else:
				# 	print('here set the entries individually depending on whether a node is at the edge or a corner or inside!') #G[i][j]['weight']=1.0/2.0;
			A=nx.to_numpy_matrix(G)
			print(A)
		zeta = la.eigvals(A)
		print('\n\nzeta',zeta)

		#This is to set the small numbers, which are not equal to zero due to the numerical calculations (lower than 1e-15), equal to zero
		for n,b in enumerate(zeta):
			# print('\n\nb[',n,']=', b, '\n\n')
			if abs(b) < 1e-15:
				zeta[n] = 0.0;
		print('\n\nzeta',zeta)
	if topology == 'global':
		zeta=np.array([-1/(Nplls-1),1]); 													#np.array([-1,Nplls-1]);
		#print('This is to test the results for the global coupling topology.')
		#time.sleep(3)
	elif topology == 'global-with-selfcoup':
		zeta=np.array([0, 1/Nplls]);														#np.array([0,Nplls]);


	print('Change here such that one can get a normalized or not normalized version of the zetas!')

#	print(zeta)
	print('\nAnalyzing the case of',topology,'coupling using the by n_k NORMALIZED adjacency matrix. The eigenvalues are zeta={',zeta,'}\n')
	return {'zeta': zeta}
# zetta=min(eigenvalzeta('1',2,1)['zet'])
# print(zetta)

# zetta=min(eigenvalzeta('1',2,1)['zet'])
# print(zetta)


# elif topology == 'square open':
# 	 for i in range(Nx):
# 		 for j in range(Ny):
# 			if ((i==0 and j==0) or (i==Nx and j==1) or (i==1 and j==Ny) or (i==Nx and j==Ny)):
# 				G[i][j]['weight']=2.0
# 				# G.add_edge(i, j, weight=1./2.)# A[i,j]=A[i,j]/2.0;
# 			# elif ((i!=1 and j==1) or (i!=Nx-1 and j==1) or (i!=1 and j==Ny) or (i!=Nx and j==Ny)):
			# 	G.remove_edge(i,j)
			# 	G.add_edge(i, j, weight=1./3.)# A[i,j]=A[i,j]/3.0;
			# else:
			# 	G.remove_edge(i,j)
			# 	G.add_edge(i, j, weight=1./4)#  A[i,j]=A[i,j]/4.0;
