using Pkg
cd(@__DIR__)
Pkg.activate("./env-hmm-j.1.10/")

using DelimitedFiles
using StatsPlots
using Turing
using Random
using Clustering
using ReverseDiff
using HiddenMarkovModels
using FillArrays
using LinearAlgebra
using LogExpFunctions
using JLD2
using PlotlyJS
using KernelDensity
using ArgParse
using Printf
Random.seed!(1)

include("./src/utils.jl")
include("./src/base_hmm.jl")


"""
    estimate_hmm(data_file::String,idx_data::Int,date_start::String,date_end::String,n_samples::Int,my_cluster_method::Symbol,n_clusters_max::Int,save_sample_dir::String,save_sample_name::String)

driver code for estimating the parameters of the hidden markov model 

args:

    data_file: csv file containing the data. the code assumes that the column names are in the first row and the firs column corresponds to the dates.
    idx_data: column index of the data
    date_start: string specifying the first date of the data file you want to use
    date_end: string specifying the last date of the data file you want to use
    n_samples: the number of samples you want to draw from the posterior
    my_cluster_method: symbol indicating whether to use k-means, k-medoids-abs or k-medoids-square
    n_clusters_max: maximum number of clusters. the optimal number of clusters will be based on the silhouette scores of the clustering results with k=2 to k=n_clusters_max clusters
    save_sample_dir: directory where the results will be saved
    save_sample_name: file name for the saved results

returns:

    nothing

mutates:

    saves sample_results, obs_data, data_dates and clusters as save_sample_dir/save_sample_name.jld2
    
"""
function estimate_hmm(data_file::String,idx_data::Int,date_start::String,date_end::String,n_samples::Int,my_cluster_method::Symbol,n_clusters_max::Int,save_sample_dir::String,save_sample_name::String)

    # load the data file
    data=readdlm(data_file,',')

    # get the dates and data
    data_dates_tmp=data[2:end,1]
    data_vals_tmp=float.(data[2:end,idx_data])

    # get the index of the start and end date of the subsample
    idx_start=findfirst(data_dates_tmp.==date_start)
    idx_end=findfirst(data_dates_tmp.==date_end)

    # get the dates and the values for the subsample
    data_dates=data_dates_tmp[idx_start:idx_end]
    data_vals=data_vals_tmp[idx_start:idx_end]
    
    # get the number of observations
    n_obs=length(data_vals)

    # allocate memory for storing the silhouette scores
    clusters_silhouettes=zeros(n_clusters_max-1,)
    
    # get cluster means and standard deviations based either on kmeans clustering or kmedoids clustering
    if my_cluster_method==:kmeans
        # use k means to set prior means and variances
        for idx_s in 2:n_clusters_max
            clusters_vals=kmeans(data_vals',idx_s)
            
            clusters_silhouettes[idx_s-1]=clustering_quality(data_vals',clusters_vals, quality_index=:silhouettes)
        end

        # optimal number of clusters based on silhouette score
        n_clusters=argmax(clusters_silhouettes)+1
        clusters_vals=kmeans(data_vals',n_clusters)
            
        # get within cluster standard deviations
        clusters_stdevs=zeros(n_clusters,)
        for idx_c in 1:n_clusters
            clusters_stdevs[idx_c]=std(data_vals[clusters_vals.assignments.==idx_c])
        end
        
        clusters_means=vec(clusters_vals.centers)
    else
        # create the data distance matrix
        if my_cluster_method==:kmedoids_abs
            data_distance_matrix=abs.(data_vals.-data_vals')
        elseif my_cluster_method==:kmedoids_sq
            data_distance_matrix=(data_vals.-data_vals').^2
        end
        # use k means to set prior means and variances
        for idx_s in 2:n_clusters_max
            clusters_vals=kmedoids(data_distance_matrix,idx_s)
            
            clusters_silhouettes[idx_s-1]=clustering_quality(clusters_vals.assignments,data_distance_matrix,quality_index=:silhouettes)
        end

        # optimal number of clusters based on silhouette score
        n_clusters=argmax(clusters_silhouettes)+1
        clusters_vals=kmedoids(data_distance_matrix,n_clusters)
            

        # sort the clusters in terms of increasing medoid
        my_medoids=data_vals[clusters_vals.medoids]
        idx_perm=sortperm(my_medoids)

        # sort the results
        clusters_vals.medoids[:]=clusters_vals.medoids[idx_perm]
        clusters_vals.counts[:]=clusters_vals.counts[idx_perm]
        for idx_o in 1:n_obs
            clusters_vals.assignments[idx_o]=idx_perm[clusters_vals.assignments[idx_o]]
        end

        # get within cluster standard deviations
        clusters_stdevs=zeros(n_clusters,)
        for idx_c in 1:n_clusters
            clusters_stdevs[idx_c]=std(data_vals[clusters_vals.assignments.==idx_c])
        end
        
        # get the cluster centers
        clusters_means=vec(data_vals[clusters_vals.medoids])
    end

    obs_data=deepcopy(data_vals)

    chain = sample(bayes_hmm(obs_data,n_clusters,clusters_means,clusters_stdevs,true),NUTS(n_samples,0.65,max_depth=5), n_samples)

    JLD2.save(save_sample_dir*save_sample_name*".jld2",Dict("sample_results"=>chain,"data"=>obs_data,"dates"=>data_dates,"clusters"=>clusters_vals))

    return nothing
end

function main(args)

    s = ArgParseSettings(description = "Sampling from posterior using hmc")

    @add_arg_table! s begin
        "--data_file"
        "--idx_data"
        "--date_start"
        "--date_end"
        "--n_samples"
        "--my_cluster_method"
        "--n_clusters_max"
        "--save_sample_dir"
        "--save_sample_name"
    end

    parsed_args=parse_args(args,s)
    data_file=parsed_args["data_file"]
    idx_data=parse(Int,parsed_args["idx_data"])
    date_start=parsed_args["date_start"]
    date_end=parsed_args["date_end"]
    n_samples=parse(Int,parsed_args["n_samples"])
    if parsed_args["my_cluster_method"]=="kmeans"
        my_cluster_method=:kmeans
    elseif parsed_args["my_cluster_method"]=="kmedoids_abs"
        my_cluster_method=:kmedoids_abs
    elseif parsed_args["my_cluster_method"]=="kmedoids_sq"
        my_cluster_method=:kmedoids_sq
    else
        Printf.@printf("my_cluster_method must be either kmeans, kmedoids_abs or kmedoids_sq. you entered %s\n",parsed_args["my_cluster_method"])
        return nothing
    end
    n_clusters_max=parse(Int,parsed_args["n_clusters_max"])
    save_sample_dir=parsed_args["save_sample_dir"]
    save_sample_name=parsed_args["save_sample_name"]


    Printf.@printf("your julia version is %s\n",VERSION)
    
    estimate_hmm(data_file,idx_data,date_start,date_end,n_samples,my_cluster_method,n_clusters_max,save_sample_dir,save_sample_name)

    return nothing
end

main(ARGS)
