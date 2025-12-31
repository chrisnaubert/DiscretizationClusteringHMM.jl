"""

args:

    chain_load_name: path to the chain
    table_save_dir: directory to where the tables will be saved
    table_save_name: prefix for the tables. the tables will be saved as 
        - table_save_dir/table_save_name-posterior.txt

retunrs:

    nothing
"""
function create_posterior_table(chain_load_name::String,table_save_dir::String,table_save_name::String)
    
    # create the directory where the tables will be saved if it doesn't exist
    if !isdir(table_save_dir)
        mkdir(table_save_dir)
    end

    # load the results
    my_results=JLD2.load(chain_load_name)

    # extract the different objects from the results
    chain=my_results["sample_results"]
    clusters_vals=my_results["clusters"]
    data_dates=my_results["dates"]
    obs_data=my_results["data"]

    # get the number of clusters
    n_clusters=length(clusters_vals.counts)

    # get number of samples
    n_samples,_,_=size(chain)

    # number of parameters
    n_pars=2*n_clusters+n_clusters^2

    # compute the ess share, mean, median, hpd intervals and rhat
    posterior_ess_share=(ess(chain,kind=:tail).:nt.:ess[1:n_pars])./(n_samples)
    posterior_mean=mean(chain).:nt.:mean[1:n_pars]
    posterior_median=quantile(chain,q=[0.5]).nt.var"50.0%"[1:n_pars]
    posterior_hpd=hpd(chain)
    posterior_lower_hpd=posterior_hpd.:nt.:lower[1:n_pars]
    posterior_upper_hpd=posterior_hpd.:nt.:upper[1:n_pars]
    posterior_rhat=rhat(chain).:nt.:rhat[1:n_pars]

   
    table=map(x->@sprintf("%0.4f",x),posterior_mean).*" & ".*map(x->@sprintf("%0.4f",x),posterior_median).*" & [".*map(x->@sprintf("%0.4f",x),posterior_lower_hpd).*",".*map(x->@sprintf("%0.4f",x),posterior_upper_hpd).*"] & ".*map(x->@sprintf("%0.4f",x),(posterior_ess_share)*100.0).*" & ".*map(x->@sprintf("%0.4f",x),(posterior_rhat))

   
    open(table_save_dir*table_save_name*"-posterior-results.txt", "w") do io
        writedlm(io, table)
    end
 

    return nothing
end