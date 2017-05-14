workspace()

using HDF5
using PyPlot

c = h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/rbm_input_new_new.h5", "r") do file
	    read(file, "a")
	end
p=c'
d= zeros(length(p[:,1]),length(p[:,1]))


	for i in 1:length(p[:,1]) #5000 examples
		for j in 1:length(p[:,1]) #5000 examples

                 		 d[i,j] =(sum(abs2(p[i,:]-p[j,:]))/700)

		end
	end

#println("distance", d)



h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/dist_mat_new_new.h5", "w") do file
    write(file, "a", d)  # alternatively, say "@write file a"
end


m=LowerTriangular(d)
lower_triangular_matrix = convert(Array{Float64,2}, m)
s=vec(lower_triangular_matrix)
x = filter(s->s!=0,s)





nbins = 200 # Number of bins

##########
#  Plot  #
##########
fig = figure("pyplot_histogram",figsize=(10,10)) # Not strictly required
ax = axes() # Not strictly required
h = plt[:hist](x,nbins) # Histogram

grid("on")
xlabel("X")
ylabel("Y")
title("Histogram")

 


