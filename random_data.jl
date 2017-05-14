workspace()

function random_data(s)

y=rand(1:length(s))
x=rand(s,(1,y))
k=1
	while sum(y) !==length(s) && k < 9      
                y=rand(1:length(s))
		x=rand(s,(1,y))
               k += 1
               println(x[k])
	end
 

end
