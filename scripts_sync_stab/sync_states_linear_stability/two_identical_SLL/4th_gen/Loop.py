import LiStaGloF_lib
from LiStaGloF_lib import K
import numpy as np
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Logo
        #self.image('logo_pb.png', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 13)
        # Move to the right
        self.cell(50)
        # Title
        self.cell(100, 10, 'Wishlist (Loop Gain and Loop Bandwidth)', 1, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

# pdf = FPDF()
pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()
pdf.set_font('Times', '', 12)

w    	  	= 2.0*np.pi*62E9; 			# intrinsic	frequency
Kvco      	= 2.0*np.pi*(0.818074E9); 	# Sensitivity of VCO
AkPD	  	= 0.162*1.0					# Peek to Peek Amplitude of the output of the PD --
GkLF		= 1.0
Gvga	  	= 2.0							# Gain of the first adder
tauf 	  	= 0.0						# tauf = sum of all processing delays in the feedback
order 	  	= 1.0
fc          =[0.01E6,0.1E6,1E6,10E6, 100E6, 1E9]
# tauc	  	= 1.0/(2.0*np.pi*100E6); 	# the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v	 	  	=  [128,256,512,1024]

f=open( 'Loop_Gain_Loop_Bandwidth.txt', 'w' )
print( '***********************************', file=f )
pdf.cell( 0, 12,'sol1=  sqrt( ( -1.0 + sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )' , 50, 100 )
pdf.cell( 0, 12,'sol2= -sqrt( ( -1.0 + sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )' , 50, 100 )
# pdf.cell(0, 12,'sol3=  sqrt( ( -1.0 - sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )=' + str(np.sqrt( ( -1.0 - np.sqrt( 1.0 + 4.0 *( K(Kvco, AkPD, GkLF,Gvga)/v[1]**2 )*( 1.0/(2.0*np.pi*fc[1])**2 ) ) ) /( 2.0*1.0/(2.0*np.pi*fc[1])**2 ) ) ) , 50, 100)
# pdf.cell(0, 12,'sol4= - sqrt( ( -1.0 - sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )=' + str(-np.sqrt( ( -1.0 - np.sqrt( 1.0 + 4.0 *( K(Kvco, AkPD, GkLF,Gvga)/v[1]**2 )*( 1.0/(2.0*np.pi*fc[1])**2 ) ) ) /( 2.0*1.0/(2.0*np.pi*fc[1])**2 ) ) ) , 50, 100)

pdf.cell( 0, 10,'**************************************************************************************', 0, 1 )
# print('sol1=  np .sqrt( ( -1.0 + np.sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )')
# print('sol2= - np .sqrt( ( -1.0 + np.sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )')
for i in range(len(v)):

    for j in range(len(fc)):

        tauc=  1.0/(2.0*np.pi*fc[j])
        alphamax= ( K(Kvco, AkPD, GkLF, Gvga)/v[i] )

        LoopBand1= np.sqrt( ( -1.0 + np.sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )
        LoopBand2=-np.sqrt( ( -1.0 + np.sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )
        LoopBand3= np.sqrt( ( -1.0 - np.sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )
        LoopBand4=-np.sqrt( ( -1.0 - np.sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )

        print( 'sol3=',LoopBand3 )
        print( 'sol4=',LoopBand4 )
        # s=str('v=',v[i],'wc=', wc[j], 'Steady State Loop Gain=', (alphamax/(2.0*np.pi)),'Loop Bandwidth=', LoopBand/(2.0*np.pi))
        # f.write(s)
        pdf.cell( 0, 8,'sol1   '  'v=' + str(v[i]) + '    fc= ' + str("{:.0e}".format(fc[j])) + 'Hz,    ' +   'Steady State Loop Gain=' + str("{:.4e}".format(alphamax/(2.0*np.pi))) + ' Hz,    '+ 'Loop Bandwidth=' + str("{:.4e}".format(LoopBand1/(2.0*np.pi)))+ 'Hz' , 50, 100)
        print( 'v=',v[i],', ', 'fc= ', "{:.0e}".format(fc[j]),' Hz,', '  Steady State Loop Gain= ', "{:.4e}".format(alphamax/(2.0*np.pi)),'Hz, ', 'Loop Bandwidth=', "{:.4e}".format(LoopBand1/(2.0*np.pi)),'Hz', file=f)

        pdf.cell( 0, 8,'sol2   ' 'v=' + str(v[i]) + '    fc= ' + str("{:.0e}".format(fc[j])) + 'Hz,    ' +   'Steady State Loop Gain=' + str("{:.4e}".format(alphamax/(2.0*np.pi))) + ' Hz,    '+ 'Loop Bandwidth=' + str("{:.4e}".format(LoopBand2/(2.0*np.pi)))+ 'Hz' , 50, 100)
        print( 'v=',v[i],', ', 'fc= ', "{:.0e}".format(fc[j]),' Hz,', '  Steady State Loop Gain= ', "{:.4e}".format(alphamax/(2.0*np.pi)),'Hz, ', 'Loop Bandwidth=', "{:.4e}".format(LoopBand2/(2.0*np.pi)),'Hz', file=f)

        # pdf.cell(0, 8, 'v=' + str(v[i]) + '    fc= ' + str("{:.0e}".format(fc[j])) + 'Hz,    ' +   'Steady State Loop Gain=' + str("{:.4e}".format(alphamax/(2.0*np.pi))) + ' Hz,    '+ 'Loop Bandwidth=' + str("{:.4e}".format(LoopBand3/(2.0*np.pi)))+ 'Hz' , 50, 100)
        # print('v=',v[i],', ', 'fc= ', "{:.0e}".format(fc[j]),' Hz,', '  Steady State Loop Gain= ', "{:.4e}".format(alphamax/(2.0*np.pi)),'Hz, ', 'Loop Bandwidth=', "{:.4e}".format(LoopBand3/(2.0*np.pi)),'Hz', file=f)

        # pdf.cell(0, 8, 'v=' + str(v[i]) + '    fc= ' + str("{:.0e}".format(fc[j])) + 'Hz,    ' +   'Steady State Loop Gain=' + str("{:.4e}".format(alphamax/(2.0*np.pi))) + ' Hz,    '+ 'Loop Bandwidth=' + str("{:.4e}".format(LoopBand4/(2.0*np.pi)))+ 'Hz' , 50, 100)

        # pdf.cell(0, 10, 'fc= ' + str("{:.0e}".format(fc[j])) +'Hz', 0, 1)

        # print('v=',v[i],', ', 'fc= ', "{:.0e}".format(fc[j]),' Hz,', '  Steady State Loop Gain= ', "{:.4e}".format(alphamax/(2.0*np.pi)),'Hz, ', 'Loop Bandwidth=', "{:.4e}".format(LoopBand4/(2.0*np.pi)),'Hz', file=f)
    pdf.cell( 0, 10,'                                                        ***********************************', 0,1)
    print( '***********************************', file=f )
    # print('v=',v[i],'wc=', "{:.0e}".format(wc[j]), 'Steady State Loop Gain=', "{:.4e}".format(alphamax/(2.0*np.pi)),'Loop Bandwidth=', "{:.4e}".format(LoopBand/(2.0*np.pi)), file=f)

f.close()
print( '***********************************' )
print( 'sol1=  np .sqrt( ( -1.0 + np.sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )' )
print( 'sol2= - np .sqrt( ( -1.0 + np.sqrt( 1.0 + 4.0 *( alphamax**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )' )
#
# for i in range(1, 41):
#     pdf.cell(0, 10, 'Printing line number ' + str(i), 0, 1)
pdf.output( 'Loop_Gain_Loop_Bandwidth.pdf', 'F' )
