      # Gnuplot script file for plotting data in file "force.dat"
      # This file is called   force.p
      set   autoscale                        # scale axes automatically
     set key at 0.11,-0.2,0
     set key font ",12"
#
      unset log                              # remove any log-scaling
      unset label                            # remove any previous labels
      set xtic auto                          # set xtics automatically
      set ytic auto                          # set ytics automatically
      set title "Lyapunov Spectrum" font "Times-Roman,15"
      set xlabel "Vfine " font "Times-Roman,14"
      set ylabel "Kvco" font "Times-Roman,17"
 
	#h(x)=Kvco_fine_peak*( -(x)**2 + a) 
	h(x)= a+b*erf(x-c)
	#Kvco_fine_peak*( 1 /( (x-1)**2 + 1) )

	#f(x) = 21.5e9+2.8e9*sqrt(x)
	fit[x=0:4] h(x) "Foutdata1.txt" using 1:2 via a,b,c

       # plot "Sensitivity_fine.txt" using 2:3 title "" w l lt rgb "red"

	#pl [t=-5:3] 'Tuning_Range_fine.txt' u 2:3, f(x)

         
	
	 

