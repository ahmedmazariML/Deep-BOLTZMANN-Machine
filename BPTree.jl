
using Graphs

type BPTree
  gr::SimpleGraph
  n_nodes::Int
  deg::Int
  nb_edges::Int
  match_edges::Array{Float64,1}
  BPmsg::Array{Array{Float64,2},1}
  Mh::Array{Array{Float64,2},1}
  Vb::Array{Array{Float64,2},1}
  c_gr::SimpleGraph
end

function BPTree(nb_nodes::Int,conn::Int)
  BPTree(caylay_tree(nb_nodes,conn),
        nb_nodes,conn,0,
        Array(Float64,0),
        Array(Array{Float64,2},0),
        Array(Array{Float64,2},0),
        Array(Array{Float64,2},0),
        simple_graph(0,is_directed=true)
         )
end

function BP_iter_fix(bp::BPTree)
  center,inc = BP_leaves_to_center(bp)
  BP_center_to_leaves(bp,inc,center)
end

function BP_iter_marg(bp::BPTree)
  print("todo")
end

function caylay_tree(n_nodes,conn; seed=1234)
  ## srand(seed)
  list_nodes = collect(1) # start a list with the first node
  rem_nodes = collect(2:n_nodes) # remaining nodes to put in the graph
  lab_nodes = collect(1:n_nodes) # labels

  g = simple_graph(n_nodes,is_directed=true) # create an undirected graph with no edges
  notfinished=true
  i = 1

  while(notfinished)
    current_node = list_nodes[i] # take current node, and add c neightbours
    for j in collect(1:conn)
      if(size(rem_nodes)[1]!=0) # if there are remaining nodes
        rdm_node = 1; ## rand(1:size(rem_nodes)[1]) # take a node at random from the remaining ones
        add_edge!(g,rem_nodes[rdm_node],current_node)
        push!(list_nodes,rem_nodes[rdm_node])
        deleteat!(rem_nodes,rdm_node)
      end
    end
    i += 1
    if(size(rem_nodes)[1]==0) # no more nodes
      notfinished=false
    end
  end
  return g
end

function copy_gr(gr)
  c_gr = simple_graph(num_vertices(gr),is_directed=is_directed(gr))
  for e in edges(gr)
    add_edge!(c_gr,e)
  end
  return c_gr
end



function find_leaf(gr)
  leaf_list = Int[]
  edges_list = Edge{Int}[]
  for v in vertices(gr)
     if(in_degree(v,gr) == 0)
      push!(leaf_list,v)
      push!(edges_list,out_edges(v,gr)[1])
    end
  end
  return leaf_list,edges_list
end


function pnorm(v)
  # norm=sum(v)
  # return v/sum(v)
  z = cumsum(v,2)[:,2]
  return v ./ z, z
end


function BP_update(bp::BPTree,ind,upd_edge)
  src_site = source(upd_edge,bp.c_gr)
  targ_site = target(upd_edge,bp.c_gr)
  bias_k = ones(size(bp.Vb[src_site]))

  for ne in in_edges(src_site,bp.c_gr)
    if(source(ne,bp.c_gr)!=targ_site)
      bias_k .*= bp.BPmsg[edge_index(ne,bp.c_gr)] * bp.Mh[edge_index(ne,bp.c_gr)]
    end
  end
  bp.BPmsg[ind],_ = pnorm(bias_k .* bp.Vb[src_site])
  ## BP_msg[ind] = pnorm(bias_k .* [lb[site],1.0/lb[site]])
end

function BP_marg(bp::BPTree,ind)
  bias_k = ones(size(bp.Vb[ind]))
  for ne in in_edges(ind,bp.c_gr)
    bias_k .*= bp.BPmsg[edge_index(ne,bp.c_gr)] * bp.Mh[edge_index(ne,bp.c_gr)]
  end
  # println(pnorm(bias_k))
  return pnorm(bias_k .* bp.Vb[ind])
  ## return pnorm(bias_k .* [lb[site],1.0/lb[site]])
end


function BP_leaves_to_center(bp::BPTree)
  nedge_list = Edge{Int}[]
  tot_inc = zeros(num_vertices(bp.c_gr))
  leaf_edge_list = find_leaf(bp.c_gr)[2]
  center = -1

  # gogin through the leaves
  for e in leaf_edge_list # first doing the first unbiased msg
    ind=edge_index(e,bp.c_gr) # index of the edge to be updated
    # msg = pnorm(Vb[source(e,gr)]) # test random init
    # msg = pnorm(rand(2))
    bp.BPmsg[ind],_ = pnorm(bp.Vb[source(e,bp.c_gr)])  ## init BP msg

    # check for adding next edges
    targ=target(e,bp.c_gr) # find target and upd/check for incoming edges
    tot_inc[targ] += 1
    ## print(e," ",targ,"\n")
    if(tot_inc[targ]==in_degree(targ,bp.c_gr))
      ## print("UPD ",targ," ",out_edges(targ,gr)[1],"\n")
      push!(nedge_list,out_edges(targ,bp.c_gr)[1])
    end
  end


  # going toward the center
  for e in nedge_list # doing the rest
    ## println("size of nedge ",size(nedge_list))
    ## print(e,"\n")
    ind=edge_index(e,bp.c_gr) # edge index
    BP_update(bp,ind,e)

    # check for edges to be updated next
    targ=target(e,bp.c_gr)
    tot_inc[targ] += 1
    if(tot_inc[targ]==in_degree(targ,bp.c_gr))
        ## print(targ,"\n")
        if(out_degree(targ,bp.c_gr)>0) # if not the center
            push!(nedge_list,out_edges(targ,bp.c_gr)[1])
            ## print(out_edges(targ,gr)[1])
        else # we are arrived at the center
          center = targ
        #  for i_e in in_edges(targ,gr)
        #    add_edge!(gr,targ,source(i_e,gr))
        #    Mh[num_edges(gr)] = Mh[edge_index(i_e)]'
            ## # creating and index for edges
            ## match_edges[edge_index(i_e)] = num_edges(gr)
            ## match_edges[num_edges(gr)] = edge_index(i_e)
        #  end
        #  for i_e in out_edges(targ,gr)
        #    push!(nedge_list_back, i_e)
        #  end
        end
    end
  end
  return center,tot_inc
end



function BP_fix_center_to_leaves(bp::BPTree, site, hid)
  done = zeros(num_vertices(bp.c_gr))
  list_site = zeros(Int,0)
  # choosing a state for the center node
  BP_fix_site_and_upd(bp,site,hid)
  done[site] = 1
  push!(list_site,site);

  for s in list_site
    for e in out_edges(s,bp.c_gr)
      neigh = target(e,bp.c_gr)
      # println(s," ",neigh)
      if(done[neigh]==0)
        # println("doing ", neigh)
        #for ne in in_edges(neigh,gr)
        #  if(ne!=e)
        #    println("see ",neigh," ",source(ne,gr))
        #    add_edge!(gr,neigh,source(ne,gr))
        #  end
        #end
        BP_fix_site_and_upd(bp,neigh,hid)

        done[neigh] = 1
        if(size(in_edges(neigh,bp.c_gr))[1]>1)
          push!(list_site,neigh)
        end
      end
    end
  end
end

function BP_center_to_leaves(bp::BPTree, center, tot_inc)
  ## println("number of speading neighb",size(nedge_list))
  nedge_list = Edge{Int}[]

  # add outgoing edges from the center!
  for i_e in in_edges(center,bp.c_gr)
    add_edge!(bp.c_gr,target(i_e,bp.c_gr),source(i_e,bp.c_gr))
    bp.Mh[num_edges(bp.c_gr)] = bp.Mh[edge_index(i_e)]'

    # creating and index for edges
    bp.match_edges[edge_index(i_e,bp.c_gr)] = num_edges(bp.c_gr)
    bp.match_edges[num_edges(bp.c_gr)] = edge_index(i_e,bp.c_gr)
  end
  for i_e in out_edges(center,bp.c_gr)
    push!(nedge_list, i_e)
  end

  # starting from the center, going to the leaves
  for e in nedge_list
    # println(e,"\n")
    ind=edge_index(e,bp.c_gr)
    BP_update(bp,ind,e)

    # check for adding other edges
    targ=target(e,bp.c_gr)
    tot_inc[targ] += 1
    if(tot_inc[targ]==in_degree(targ,bp.c_gr))
        ## print(targ,"\n")
        if(in_degree(targ,bp.c_gr)!=out_degree(targ,bp.c_gr)) # not yet on a leaf
          for ne in in_edges(targ,bp.c_gr)
            #print(ne," ",e,"\n")
            if(ne!=e)
             #print("pouet\n")
             add_edge!(bp.c_gr,targ,source(ne,bp.c_gr))
             bp.Mh[num_edges(bp.c_gr)] = bp.Mh[edge_index(ne,bp.c_gr)]'
             ## # creating and index for edges
             bp.match_edges[edge_index(ne,bp.c_gr)] = num_edges(bp.c_gr)
             bp.match_edges[num_edges(bp.c_gr)] = edge_index(ne,bp.c_gr)
            end
          end
          #print("\n\n")
          for ne in out_edges(targ,bp.c_gr)
            #print(ne,"\n")
            if(target(ne,bp.c_gr)!=source(e,bp.c_gr))
              ## print(ne,"\n")
              push!(nedge_list, ne)
            end
          end
        end
    end
  end
end


function BP_fix_site_and_upd(bp::BPTree,site, hid)
  psi,_ = BP_marg(bp::BPTree,site)
  #if( (site>=1) & (site<=10))
  #  println(site," ",psi[1,2])
  #end
  # project the probability
  # generate (n_samples) random number
  rdm = rand(size(psi,1));
  ## psi = !(psi .< reshape([rdm;1-rdm],size(psi)))
  psi = (reshape([rdm;1-rdm],size(psi)) .< psi)*1

  # assign the node a state
  hid[site,:] = psi[:,2]
  #println("State ",hid[site])

  #println("poeut")
  #println(site," ",out_edges(site,bp.c_gr))
  out_e = out_edges(site,bp.c_gr)
  if(size(out_e)[1]==0) # no outgoing link = center node
    targ=-1
  else
    targ = target(out_e[1],bp.c_gr)
  end

  # add missing out_edges
  for e in in_edges(site,bp.c_gr)
    #println("e ",e,  out_e)
    if(source(e)!=targ)
      #println("src-targ ",site," ",source(e,bp.c_gr))
      add_edge!(bp.c_gr,site,source(e,bp.c_gr))
      bp.Mh[num_edges(bp.c_gr)] = bp.Mh[edge_index(e)]'
      #println("CHECK1: ",e," ",num_edges(bp.c_gr)," ",bp.Mh[num_edges(bp.c_gr)])
    end
  end

  for e in out_edges(site,bp.c_gr)
    #println("CHECK2: ",e," ",edge_index(e,bp.c_gr)," ",bp.Mh[edge_index(e,bp.c_gr)])
    bp.BPmsg[edge_index(e,bp.c_gr)] = psi
  end
end

function BP_compute_2pmarg(bp::BPTree)

 marg2p_gr = zeros(num_edges(bp.gr),size(bp.BPmsg[1],1)) #Array{Float64,1}[]
  Z_tot = zeros(num_edges(bp.gr),size(bp.BPmsg[1],1))
  ## println(size(marg2p_gr))
  for e in edges(bp.gr)
    ind1 = Int(edge_index(e,bp.gr))
    ind2 = Int(bp.match_edges[ind1])
    site1 = source(e,bp.gr)
    site2 = target(e,bp.gr)
    Z = zeros(Float64,0)

    # TODO : find a better way to write this !
    for s in collect(1:size(bp.BPmsg[ind1])[1])
      push!(Z,(bp.BPmsg[ind1][s,:] * bp.Mh[ind1] * bp.BPmsg[ind2][s,:]')[1] )
    end
    ## println(edge_index(e,bp.gr))
    marg2p_gr[edge_index(e,bp.gr),:] = (bp.BPmsg[ind1][:,2] .* bp.BPmsg[ind2][:,2] .* bp.Mh[ind1][2,2]) ./ Z
    Z_tot[ind1,:] = Z[:]
    # push!(marg2p_gr,(bp.BPmsg[ind1][:,2] .* bp.BPmsg[ind2][:,2]) ./ Z )
  end

#println("marg2p_gr",size(marg2p_gr))
  return marg2p_gr,Z_tot
end

function BP_compute_2pmarg_full(bp::BPTree)
  ## marg2p_gr = zeros(num_edges(bp.gr),size(bp.BPmsg[1],1)) #Array{Float64,1}[]
  marg2p_gr = Array(Array{Float64,2},num_edges(bp.gr))
  Z_tot = zeros(num_edges(bp.gr),size(bp.BPmsg[1],1))
  ## println(size(marg2p_gr))
  for e in edges(bp.gr)
    ind1 = Int(edge_index(e,bp.gr))
    ind2 = Int(bp.match_edges[ind1])
    site1 = source(e,bp.gr)
    site2 = target(e,bp.gr)
    Z = zeros(Float64,0)
    ei = edge_index(e,bp.gr)
    marg2p_gr[ei] = zeros(size(bp.BPmsg[ind1],1),4)

    # TODO : find a better way to write this !
    for s in collect(1:size(bp.BPmsg[ind1])[1])
      Z = (bp.BPmsg[ind1][s,:] * bp.Mh[ind1] * bp.BPmsg[ind2][s,:]')[1]
      #b = (bp.BPmsg[ind1][s,:]' * bp.BPmsg[ind2][s,:])
      #println(b)
      #println(size(b))
      marg2p_gr[ei][s,:] = (((bp.BPmsg[ind1][s,:]' * bp.BPmsg[ind2][s,:]) .* bp.Mh[ind1])./ Z)
      #marg2p_gr[ei][s,:] = reshape( (((bp.BPmsg[ind1][s,:]' * bp.BPmsg[ind2][s,:]) .* bp.Mh[ind1])./ Z),(4,1))
    end
  end
  return marg2p_gr
end

function BP_compute_marg(bp::BPTree)
  marg_gr = Array{Float64,2}[]
  for n in vertices(bp.c_gr)
    marg,_ = BP_marg(bp,n)
    push!(marg_gr,marg)
    ## marg_gr[n] = marg(n,gr,BP_msg,Mh)
  end
  return marg_gr
end

function BP_compute_fnrg(bp::BPTree)
  fsite = zeros(size(bp.BPmsg[1],1))
  for n in vertices(bp.c_gr)
    _,zsite = BP_marg(bp,n)
    fsite += log(zsite)
  end

  _,flink = BP_compute_2pmarg(bp)
  return -(fsite-sum(log(flink),1)' ) #/bp.n_nodes
end

function BP_init_sampling(bp::BPTree, hbias, J, vis)
  resize!(bp.BPmsg,2*num_edges(bp.gr))
  resize!(bp.match_edges,2*num_edges(bp.gr))
  bp.c_gr = copy_gr(bp.gr)

  local_b = vis .+ hbias
  resize!(bp.Mh,2*num_edges(bp.gr))
  for e in edges(bp.gr)
    i_e = edge_index(e,bp.gr)
    Jss = J[i_e]
    bp.Mh[i_e] = [ 1 1 ; 1 exp(Jss) ]
  end

  resize!(bp.Vb,num_vertices(bp.gr))
  for n in vertices(bp.gr)
    bp.Vb[n] = [exp(-local_b[n,:]*0);exp(local_b[n,:])]'
  end
end

function MC_sampling(bp::BPTree, t_therm, t_msr)
  ## RDM Init
  ns = size(bp.Vb[1],1)
  site = bitrand(bp.n_nodes,ns)*1

  # Thermalize
  for i in collect(1:t_therm)
    rdmsite = rand(1:bp.n_nodes,bp.n_nodes)
    for j in rdmsite
      for s in 1:ns
        hi_loc = log(bp.Vb[j][s,2])
        for k in out_edges(j,bp.gr)
          hi_loc += log(bp.Mh[edge_index(k,bp.gr)][2,2])*site[target(k,bp.gr)]
        end
        for k in in_edges(j,bp.gr)
          hi_loc += log(bp.Mh[edge_index(k,bp.gr)][2,2])*site[source(k,bp.gr)]
        end
        hi_loc *= (2*site[j,s]-1)

        if(hi_loc < 0)
          site[j,s] = !Bool(site[j,s])*1
        elseif (exp(-hi_loc) > rand())
          site[j,s] = !Bool(site[j,s])*1
        end
      end
    end
  end

  # MSR
  delta_msr = 10
  tot = 0
  Marg = zeros(bp.n_nodes,ns)
  Marg2p = zeros(bp.n_nodes*2,ns)
  for i in collect(1:t_msr)
    rdmsite = rand(1:bp.n_nodes,bp.n_nodes)
    for j in rdmsite
      for s in 1:ns
        hi_loc = log(bp.Vb[j][s,2])
        for k in out_edges(j,bp.gr)
          hi_loc += log(bp.Mh[edge_index(k,bp.gr)][2,2])*site[target(k,bp.gr)]
        end
        for k in in_edges(j,bp.gr)
          hi_loc += log(bp.Mh[edge_index(k,bp.gr)][2,2])*site[source(k,bp.gr)]
        end
        hi_loc *= (2*site[j,s]-1)

        if(hi_loc < 0)
          site[j,s] = !Bool(site[j,s])*1
        elseif (exp(-hi_loc) > rand())
          site[j,s] = !Bool(site[j,s])*1
        end
      end
    end

    if((i%delta_msr)==1)
      Marg += site
      for e in edges(bp.gr)
        ind1 = edge_index(e,bp.gr)
        ind2 = bp.match_edges[ind1]
        site1 = source(e,bp.gr)
        site2 = target(e,bp.gr)
        Marg2p[ind1,:] += site[site1] .* site[site2]
      end
      ## Marg2p += site * site'
      tot += 1
    end

  end
  println(tot)
  Marg /= tot
  Marg2p /= tot
  return Marg, Marg2p

end
