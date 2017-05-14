module Tree

import Base: start, done, next, show

export Node,
    addchild,
    addsibling,
    isroot,
    isleaf,
    lastsibling



type Node{T}
    data::T
    parent::Node{T}
    child::Node{T}
    sibling::Node{T}
    
    # Constructor for the root of the tree
    function Node(data::T)
        n = new(data)
        n.parent = n
        n.child = n
        n.sibling = n
        n
    end
    # Constructor for all others
    function Node(data::T, parent::Node)
        n = new(data, parent)
        n.child = n
        n.sibling = n
        n
    end
end
Node{T}(data::T) = Node{T}(data)
Node{T}(data::T, parent::Node{T}) = Node{T}(data, parent)

function lastsibling(sib::Node)
    newsib = sib.sibling
    while sib != newsib
        sib = newsib
        newsib = sib.sibling
    end
    sib
end

function addsibling{T}(oldersib::Node{T}, data::T)
    if oldersib.sibling != oldersib
        error("Truncation of sibling list")
    end
    youngersib = Node(data, oldersib.parent)
    oldersib.sibling = youngersib
    youngersib
end

function addchild{T}(parent::Node{T}, data::T)
    newc = Node(data, parent)
    prevc = parent.child
    if prevc == parent
        parent.child = newc
    else
        prevc = lastsibling(prevc)
        prevc.sibling = newc
    end
    newc
end

isroot(n::Node) = n == n.parent
isleaf(n::Node) = n == n.child

show(io::IO, n::Node) = print(io, n.data)


start(n::Node) = n.child
done(n::Node, state::Node) = n == state
next(n::Node, state::Node) = state, state == state.sibling ? n : state.sibling

function showedges(io::IO, parent::Node, printfunc = identity)
    str = printfunc(parent.data)
    if str != nothing
        if isleaf(parent)
            println(io, str, " has no children")
        else
            print(io, str, " has the following children: ")
            for c in parent
                print(io, printfunc(c.data), "    ")
            end
            print(io, "\n")
            for c in parent
                showedges(io, c, printfunc)
            end
        end
    end
end
showedges(parent::Node) = showedges(STDOUT, parent)

end
