import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

"""
Sukhbinder
5 April 2017

Based on:
"""

def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj

def intersection(x1,y1,x2,y2):
    """
    INTERSECTIONS Intersections of curves.
    Computes the (x,y) locations where two curves intersect.  The curves
    can be broken with NaNs or have vertical segments.
    """

    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]

def K(Kvco, AkPD, Ga1):
	return (Kvco*Ga1*AkPD)/2.0
cfDig        = lambda x: sawtooth(x,width=0.5)

def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list
# # define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
# w1  	= 2.0*np.pi*24.48E9
# w2		= 2.0*np.pi*24.60E9
#
# wmean   = (w1+w2)/2.0
# Dw		= w2-w1
# Kvco    = 2.0*np.pi*(757.64E6);
# AkPD	= 1.6
# Ga1     = 0.6
# order	= 2.0
#
# tauf    = 0.0																	# tauf = sum of all processing delays in the feedback
# tauc	= 1.0/(2.0*np.pi*1E6);
# v		= 512.0;
# c		= 3E8
# maxp 	= 3;

# w1  	= 2.0*np.pi*(1.0+0.01)#24.48E9##
# w2		= 2.0*np.pi*(1.0-0.01)#24.60E9##
#
# wmean   = (w1+w2)/2.0
# Dw		= w2-w1
# Kvco    = 2.0*np.pi*0.1#(757.64E6);
# AkPD	= 1.6
# Ga1     = 0.6
# order	= 2.0
#
# tauf    = 0.0																	# tauf = sum of all processing delays in the feedback
# tauc	= 1.0/(2.0*np.pi*0.1)#*1E6);
# v		= 1.0;

def integrationdigital(wmean,Dw, Kvco, AkPD, Ga1,tauf, v):

    # a piece of a prolate cycloid, and am going to find
    a, b = 1, 2
    tau= np.linspace(0.0, 36.788706E-8, 10000)#4.788706E-9#
    p = np.linspace(0, 5*np.pi, 10000)
    om=[];taucc=[];beta=[];
    # f1 = - Omega + wmean+Dw/2.0 + K(Kvco, AkPD, Ga1)* cfDig( (-(Omega*(tau-tauf))-beta)/v)
    # f2 = - Omega + wmean-Dw/2.0 + K(Kvco, AkPD, Ga1)* cfDig( (-(Omega*(tau-tauf))+beta)/v)
    for index in range(len(tau)):

        y1 = wmean+Dw/2.0 + K(Kvco, AkPD, Ga1)*cfDig(-p)
        x1 = y1*(tau[index]-tauf)+p*v
        y2= wmean-Dw/2.0 + K(Kvco, AkPD, Ga1)* cfDig(p)
        x2= -y2*(tau[index]-tauf)+p*v

        x,y=intersection(x1,y1,x2,y2)

        # print(len(x),len(y))
        ry =  [round(y,4) for y in y]
        rx =  [round(x,2) for x in x]
        # print(ry)
        # print(y/wmean)
        yy=list(set(ry/np.asarray(wmean)))
        yy1=Remove(ry/np.asarray(wmean))
        xx=list(set(rx))
        beta.append(xx)
        # print(len(y/wmean),y/wmean)
        # print(len(yy1),yy1)
        # print(len(yy), yy)
            # print(y/wmean)
        om.append(yy)
        # print('Omega=',om[index],'tau=',(wmean/(2.0*np.pi))*tau[index])
                # taucc.append(tau[index])
    o=np.concatenate(om, axis=0 )
    betadig=np.array(np.concatenate(beta, axis=0 ))
    o1=np.array(o)

    return {'OmegaDig': o1, 'betadig': betadig, 'Tau':tau}



        # om=x[index]
        # bet=y[index]
        # Omega[index] = x
		# beta[index]	= y

    # plt.plot(x1,y1/wmean,c='r')
    # plt.plot(x2,y2/wmean,c='g')
    # print(type(tau))

    # print(om,type(om))
# fig         = plt.figure()
# ax          = fig.add_subplot(111)
#
# plt.title(r'digital case for $\omega$=%.3f' %wmean);
# 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
#     # adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'$\bar{\omega}\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\Omega/\bar{\omega}$', fontsize=18)
# plt.plot((wmean/(2.0*np.pi))*integrationdigital(wmean,Dw, Kvco, AkPD, Ga1, v)['Tau'][np.where(integrationdigital(wmean,Dw, Kvco, AkPD, Ga1, v)['OmegaDig']!=0)], integrationdigital(wmean,Dw, Kvco, AkPD, Ga1, v)['OmegaDig'],'*k')
# # plt.plot(x,y/wmean,'*k')
# plt.show()
