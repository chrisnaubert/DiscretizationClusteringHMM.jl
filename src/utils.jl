function inverse_gamma_transform(mu, sd)
    a = mu^2 / sd^2 + 2
    b = mu * (a - 1)
    return a, b
end


"""
"""
function my_silhouettes(my_cluster,distance_matrix)
    # number of clusters
    n_clusters=length(my_cluster.:medoids)

    # number of data points
    n_data=length(my_cluster.:assignments)

    # storage for costs
    cost_matrix=zeros(n_data,n_clusters)

    # get the counts for each class
    class_counts=zeros(Int,n_clusters,)

    # compute within and across cluster costs
    for idx_i in 1:n_data

        # reset the class counts
        for idx_c in 1:n_clusters
            class_counts[idx_c]=1
        end

        for idx_j in 1:n_data
            
            # update the cost for idx_j!=idx_i
            if idx_j!=idx_i    
                # update the average cost
                cost_matrix[idx_i,my_cluster.:assignments[idx_j]]=cost_matrix[idx_i,my_cluster.:assignments[idx_j]]+(distance_matrix[idx_i,idx_j]-cost_matrix[idx_i,my_cluster.:assignments[idx_j]])/class_counts[my_cluster.:assignments[idx_j]]
                
                #increment the number of elements in class_counts
                class_counts[my_cluster.:assignments[idx_j]]+=1
            end
        end
    end

    # compute tha a(i) and b(i)
    # a(i): average cost to all other elements in my cluster
    # b(i): average cost to all elements in the next best cluster for item i
    a_i=zeros(n_data,)
    b_i=zeros(n_data,)
    s_i=zeros(n_data,)
    for idx_i in 1:n_data

        # get the assignment of data point idx_i
        idx_i_assignment=my_cluster.:assignments[idx_i]

        a_i[idx_i]=cost_matrix[idx_i,idx_i_assignment]

        # initialize the next best cost
        b_cost=Inf
        for idx_j in 1:n_clusters
            # if idx_j is not my current assignment, compare it to the best cost so far
            if idx_j!=idx_i_assignment && (cost_matrix[idx_i,idx_j]<b_cost)
                b_i[idx_i]=cost_matrix[idx_i,idx_j]
                b_cost=b_i[idx_i]
            end
        end
        s_i[idx_i]=(b_i[idx_i]-a_i[idx_i])/max(a_i[idx_i],b_i[idx_i])
    end

    # compute the average silhouette score
    silhouette_score=mean(s_i)

    return silhouette_score    
end