      # Gnuplot script file for plotting data in file "force.dat"
      # This file is called   force.p
   #   set   autoscale                        # scale axes automatically
   # set key at 23.2e9 ,2.2,0
     set key font ",21"
#
      unset log                              # remove any log-scaling
      unset label                            # remove any previous labels
      set xtic auto                          # set xtics automatically
      set ytic auto                          # set ytics automatically
      set title "Frequency Response Curve" font "Times-Roman,26"
      set xlabel "Vcontrol " font "Times-Roman,24"
      set ylabel "Frequency" font "Times-Roman,24"
 
	#h(x)=Kvco_fine_peak*( -(x)**2 + a) 
	f(x)= 21.5e9+1.8e9*sqrt(x)
	#Kvco_fine_peak*( 1 /( (x-1)**2 + 1) )

	#f(x) = 21.5e9+2.8e9*sqrt(x)
	#fit[x=-5:3] h(x) "1.txt" using 1:2 via a,b

       plot "1.txt" using 1:2 title "data", f(x) title "21.5e9+1.8e9*sqrt(x)"

	#pl [t=-5:3] 'Tuning_Range_fine.txt' u 2:3, f(x)

         
	
	 


