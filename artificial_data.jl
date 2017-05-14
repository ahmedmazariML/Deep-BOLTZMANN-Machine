workspace()

import HTreeRBM

function gen_random(m,n)

random_float = rand(m,n)
X=random_float 
X=round(Int64,X)

# Pkg.add("JLD"), Pkg.add("HDF5"), using JLD, HDF5
# store the matrix in a jld file 
file = jldopen("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/test_scr/mydata.jld", "w")
write(file, "X", X)  # alternatively, say "@write file A"
close(file)

file = jldopen("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/test_scr/mydata.jld", "r")
c = read(file, "X")
close(file)



Y=float_to-binary(X)

print("binary matrix",Y)
return Y;
 end


function float_binary(X;level=0.4)
  for i=1:length(X)
    X[i] = X[i] > level ? 1.0 : -1.0
  end
  return X
end

 random_data= gen_random(m,n)




