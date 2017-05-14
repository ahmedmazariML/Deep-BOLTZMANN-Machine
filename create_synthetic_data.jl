# This file represents the first version of artificial data under tree structure . We start with three level of abstraction 
#root, childs and  leafs that l set manually to do simple test. Later on l extend it to different level of hierarchy.

workspace()

using JLD, HDF5




type MyNode{T}
    vector::Vector{T}
    level::Int
    child :: Vector{MyNode}
    nchild :: Int
    index :: Int
end

# initialize the root 

MyNode() = MyNode(Float64[], 0, MyNode[], 0, 0)

# The length of each vector : each vector has 700 values 

s= rand(0.0:0.01:1.0,700)


# Binarize the values of s under a certain threshold 

function float_to_binary(s,level=0.4)

	  for i=1:length(s)
		s[i] = s[i] > level ? 1.0 : -1.0
		# Scale s to [0,1]
		s[i] = (s[i] + 1) / 2

	  end

  return s

end

data=float_to_binary(s)


# Set the number of childs in the second level
# To begin with, we set arbitrary Number of children nChildren = 10

nChildren = 10
# Initialize a vector with the root data or empty vector of length 700
vector = Vector()
	for i in 1:nChildren
           push!(vector, data[1:length(data)])
	end

# Set epsilon at level 1

epsilon1= 0.4

# Set alpha for level one following a uniform distribution (0,1)

rand_alpha1 = Vector()
	for i in 1:nChildren
		push!(rand_alpha1,rand(0.0:0.01:1.0,1,700))
	end


# flipping the spins 0 becomes 1 and 1 becomes 0 



for i in 1:nChildren
    for j in 1:length(vector[i])
	     if (epsilon1 < rand_alpha1[i][j])
			if (vector[i][j] == 0)
			  	vector[i][j] = 1
			else
		      		vector[i][j] = 0
			end
	     end
    end
end


# add_children function allows to create a tree of several levels. In the first time we limit the hierarchical level to 2 ( 0 : root, 1: childs, 2; leafs )

function add_children!(parentNode, nChildren,vector)
  

 index =1
  level = parentNode.level + 1
  for i in 1:nChildren
    child = MyNode(vector[i], level, MyNode[], 0,index)
    push!(parentNode.child, child)
    index += 1
  end
  parentNode.nchild += nChildren
 return vector
end

function node_info(node)
  println("Level    : ", node.level)
  println("nChildren: ", node.nchild)
  println("data     : ", node.vector)
  println("index ", node.index)
end

function data_node(node)
 
  println("data     : ", node.vector)
  println("index ", node.index)
end

# level 0
# create root node 
root = MyNode()

# add children to root
add_children!(root, nChildren,vector)

# create an empty list for each child, in this list we can then
# later insert leafs (grand_children)
# In the end this list will be used like this
# grand_children_list[1] = list of all grand-children of child 1.
# grand_children_list[2] = list of all grand-children of child 2.

grand_children_list = Any[MyNode[] for i in 1:nChildren]

# set the total number of grand-children (= total number of leafs)
# then create all of them and isert them into the list.

#for i in 1:10
        	# for j in 1:length(grand_children_list[i])
       
        	#	push!(leafs_child,grand_children_list[i][j].index)
        
# To begin with, we set arbitrary Number of grand_children (leafs) nGrandChildren = 5000

nGrandChildren = 5000

# Set epsilon for level two epsilon of level 2 > epsilon of level 1

epsilon2= 0.2

# For each child we have 500 leafs because we have 10 childs and 5000 leafs 

num_leafs=  nGrandChildren / nChildren

index = 0


vector_leaf = Vector()
	for i in 1:nGrandChildren
           push!(vector_leaf, data[1:length(data)]) # instead create an empty vector , to do later on
	end

	for i in 1: nChildren
                  level= 2
		for j in 1:num_leafs
			index = index +1
			vector_leaf[j] = vector[i]
			 grand_child = MyNode(vector_leaf[j], level, MyNode[], 0,index)
                         push!(grand_children_list[i], grand_child)
                       #  println("ok1")
		end
		#println("ok2")
		index=index
                #println(index)
		#println("ok3")
	end

# Set alpha for level 2 following a uniform distribution (0,1)

rand_alpha2 = Vector()
	
# flipping the spins

for i in 1:nGrandChildren
 push!(rand_alpha2, rand(0.0:0.01:1.0,1,700))
end

for i in 1:nGrandChildren
	
	
    for j in 1:length(vector_leaf[i])
              
	     if (epsilon2 < rand_alpha2[i][j])
			if (vector_leaf[i][j] == 0)
				vector_leaf[i][j] = 1
			else
				vector_leaf[i][j] = 0
			end
	      
	     end
    end
#println("ça marche")
end

# In order to fit the function run_artificial_data() where each column represents an example and a row the features , our matrix of 5000*700 becomes 700*5000

 a= zeros(length(s),0)
		for v in vector_leaf
		       a=hcat(a,v)
		end
# Store the matrix in a file 

h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/rbm_input_new_new.h5", "w") do file
    write(file, "a", a)  # alternatively, say "@write file a"
end




