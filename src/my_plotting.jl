"""
    plot_scatter_and_probs(chain_load_name,fig_save_dir,fig_save_name)

code for plotting the time series with cluster assignments and posterior probabilities of the different states

args:
    chain_load_name: path to the chain to plot
    fig_save_dir: directory to where the figures will be saved
    fig_save_name: prefix for the figures. the figures will be saved as 
        - fig_save_dir/fig_save_name-scatter.pdf
        - fig_save_dir/fig_save_name-probs.pdf
   
returns:
    nothing
"""
function plot_scatter_and_probs(chain_load_name::String,fig_save_dir::String,fig_save_name::String;my_font_size=12)

    # create the directory where the figures will be saved if it doesn't exist
    if !isdir(fig_save_dir)
        mkdir(fig_save_dir)
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


    # get the subchain for the hidden states
    chain_hidden_states=MCMCChains.group(chain, :s)
    n_samples,n_obs,_=size(chain_hidden_states)

    # get the probabilities for each state
    state_probs=zeros(n_obs,n_clusters)
    for idx_c in 1:n_clusters
        state_probs[:,idx_c]=vec(mean(chain_hidden_states.value[:,:,1].==float(idx_c),dims=1))
    end 

    my_color_palette=theme_palette(:auto)

    # create the scatter plot
    p_scatter=scatter((1:n_obs)[clusters_vals.assignments.==1], obs_data[clusters_vals.assignments.==1], label="Data Cluster 1",mode="markers",marker_color=my_color_palette[1],legend=:outertop,legend_columns=min(3,n_clusters),legendfontsize=my_font_size,xtickfontsize=my_font_size,ytickfontsize=my_font_size,bottom_margin=5.0Plots.mm)
    for idx_c in 2:n_clusters
        scatter!((1:n_obs)[clusters_vals.assignments.==idx_c], obs_data[clusters_vals.assignments.==idx_c], label="Data Cluster "*string(idx_c),mode="markers",marker_color=my_color_palette[idx_c])
    end
    plot!(xticks=(collect(1:15:n_obs),data_dates[1:15:end]),xrotation=45)

    # get the part of the chain corresponding to the discretized state values
    chain_state_values=MCMCChains.group(chain,:obs_m)

    # get the median estimates of the state values
    state_values_median=quantile(chain_state_values,q=[0.5]).nt.var"50.0%"

    # create the labels for the posterior state plots
    for idx_c in 1:n_clusters
        hline!(state_values_median[idx_c:idx_c],label="Median State Value "*string(idx_c),linewidth=4,alpha=0.5,linecolor=my_color_palette[idx_c],linestyle=:dash)
    end

    # get the median estimates of the state values
    state_values_mean=mean(chain_state_values).nt.mean

    # create the labels for the posterior state plots
    for idx_c in 1:n_clusters
        hline!(state_values_mean[idx_c:idx_c],label="Mean State Value "*string(idx_c),linewidth=4,alpha=0.5,linecolor=my_color_palette[idx_c],linestyle=:dot)
    end

    
    

    # create the bar plot of the probabilities of each state
    p_bar_labels=Array{String,2}(undef,1,n_clusters)
    for idx_c in 1:n_clusters
        p_bar_labels[1,idx_c]="Probability of State "*string(idx_c)
    end
    p_bar=groupedbar(state_probs,bar_position=:stack,linecolor = :match,legend=:outertop,legend_columns=min(3,n_clusters),label=p_bar_labels,xlims=xlims(p_scatter),legendfontsize=my_font_size,xtickfontsize=my_font_size,ytickfontsize=my_font_size,bottom_margin=5.0Plots.mm)
    plot!(xticks=(collect(1:15:n_obs),data_dates[1:15:end]),xrotation=45)

    # create a figure of the raw data
    p_data=plot((1:n_obs), obs_data, label="",markershape=:circle,markersize=4.0,legendfontsize=my_font_size,xtickfontsize=my_font_size,ytickfontsize=my_font_size,bottom_margin=5.0Plots.mm,linewidth=2.0,xlabel="Date",ylabel="Index Value")
    plot!(xticks=(collect(1:15:n_obs),data_dates[1:15:end]),xrotation=45)


    # save the scatter plot
    savefig(p_scatter,fig_save_dir*fig_save_name*"-scatter.pdf")

    # save the bar plot
    savefig(p_bar,fig_save_dir*fig_save_name*"-probs.pdf")

    # save raw data plot
    savefig(p_data,fig_save_dir*fig_save_name*"-data.pdf")

    return nothing
end

"""

    plot_simulated_series(chain_load_name::String,fig_save_dir::String,fig_save_name::String,t_sim::Int;my_font_size=12)

code for plotting a simulated series using the estimated hidden markov model. the code creates a time series plot of the simulated states and a kde comparison plot between the simulated observables and the data

args:
    chain_load_name: path to the chain to plot
    fig_save_dir: directory to where the figures will be saved
    fig_save_name: prefix for the figures. the figures will be saved as 
        - fig_save_dir/fig_save_name-simulated-series.pdf
        - fig_save_dir/fig_save_name-density-comparison.pdf
    t_sim: length of the simulation

returns:
    nothing
"""
function plot_simulated_series(chain_load_name::String,fig_save_dir::String,fig_save_name::String,t_sim::Int;my_font_size=12)
   
    # create the directory where the figures will be saved if it doesn't exist
    if !isdir(fig_save_dir)
        mkdir(fig_save_dir)
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

    # get the part of the chain corresponding to the discretized state values
    chain_state_values=MCMCChains.group(chain,:obs_m)
    
    # get the median estimates of the state values
    state_values=quantile(chain_state_values,q=[0.5]).nt.var"50.0%"

    # get the part of the chain corresponding to the state noise
    chain_state_noise=MCMCChains.group(chain,:obs_s)
    
    # get the median estimates of the state values
    state_noise=quantile(chain_state_noise,q=[0.5]).nt.var"50.0%"

    # get the part of the chain corresponding to the transition probabilities
    chain_transition_probs=MCMCChains.group(chain,:T)

    # get the median estimates of the transition probabilities
    transition_probs=quantile(chain_transition_probs,q=[0.5]).nt.var"50.0%"

    # adjoint of the transition matrix
    transition_matrix_adj=reshape(transition_probs,(n_clusters,n_clusters))
 
    # compute the eigen values and eigen vectors
    eigs_transition_matrix_adj=eigen(transition_matrix_adj)

    # find the index of the unit eigenvalue
    idx_unit_eig=findfirst(eigs_transition_matrix_adj.values.==1.0)

    # get the corresponding eigen vector
    unnormalized_stationary_dist=eigs_transition_matrix_adj.vectors[:,idx_unit_eig]

    # compute the normalized distribution
    stationary_dist=unnormalized_stationary_dist./sum(unnormalized_stationary_dist)

    # allocate memory for the simulation
    sim_states=zeros(t_sim,)

    # allocate memory for the simulated observations
    sim_obs=zeros(t_sim,)

    # draw the initial state
    idx_state=rand(Categorical(stationary_dist))

    # draw the innovations
    innovs=randn(t_sim,)

    # store the initial state value
    sim_states[1]=state_values[idx_state]
    sim_obs[1]=state_values[idx_state]+state_noise[idx_state]*innovs[1]

    for t in 2:t_sim
        # draw the new index
        idx_state=rand(Categorical(transition_matrix_adj[:,idx_state]))

        # store the state
        sim_states[t]=state_values[idx_state]

        # store the observation
        sim_obs[t]=state_values[idx_state]+state_noise[idx_state]*innovs[t]    
    end

    # create the plot of the simulated series
    p_simulation=plot(sim_states,linewidth=2,fontsize=my_font_size,label="",xtickfontsize=my_font_size,ytickfontsize=my_font_size,xlabel="Time",ylabel="Index Value")

    # save the simulation plot
    savefig(p_simulation,fig_save_dir*fig_save_name*"-simulated-series.pdf")

    # create the plot comparing kde of data with simulated series
    p_kde=density(obs_data,linewidth=2,fontsize=my_font_size,label="Data",xtickfontsize=my_font_size,ytickfontsize=my_font_size,xlabel="Index Value",legendfontsize=my_font_size,top_margin=10.0Plots.mm)
    density!(sim_obs,linewidth=2,fontsize=my_font_size,label="Simulated Series",xtickfontsize=my_font_size,ytickfontsize=my_font_size)

    # compute the ks_test_statistic
    ks_test_results=ApproximateTwoSampleKSTest(obs_data,sim_obs)

    # get the p value
    ks_test_p_value=HypothesisTests.pvalue(ks_test_results)
    ks_test_string="K-S Test P-value: "*string(round(ks_test_p_value,digits=5))
    annotate!((0.5,1.1),ks_test_string)
    # save the simulation plot
    savefig(p_kde,fig_save_dir*fig_save_name*"-density-comparison.pdf")
    

    return nothing
end

"""
    plot_trace_and_density(chain_load_name::String,fig_save_dir::String,fig_save_name::String;my_font_size=12,linewidth=2.0,right_margin=5.0Plots.mm)

create trace and density plots for the estimated transition probabilities, state means and state standard deviations

args: 
    chain_load_name: path to the chain to plot
    fig_save_dir: directory to where the figures will be saved
    fig_save_name: prefix for the figures. the figures will be saved as 
        - fig_save_dir/fig_save_name-[parameter name]-density.pdf
        - fig_save_dir/fig_save_name-[parameter name]-trace.pdf

returns:
    nothing

"""
function plot_trace_and_density(chain_load_name::String,fig_save_dir::String,fig_save_name::String;my_font_size=12,linewidth=2.0,right_margin=5.0Plots.mm)

    # load the results
    my_results=JLD2.load(chain_load_name)

    # extract the different objects from the results
    chain=my_results["sample_results"]

    # get the names of the obs_m, obs_s and T parameters
    names_obs_m=namesingroup(chain,:obs_m)
    names_obs_s=namesingroup(chain,:obs_s)
    names_T=namesingroup(chain,:T)

    # get the chain corresponding to the parameters
    chain_params=chain[reduce(vcat,(names_obs_m,names_obs_s,names_T))]

    # replace spaces, brackets and commas in names
    names_all=replace.(replace.(replace.(string.(chain_params.name_map.parameters),"["=>"_"),"]"=>""),", "=>"_")

    
    # get the number of parameters
    n_params=length(names_all)

    # create hpd intervals from the chain
    chain_hpd=hpd(chain_params)
    lower_hpd=chain_hpd.:nt.:lower
    upper_hpd=chain_hpd.:nt.:upper

    # create the directory where the figures will be saved if it doesn't exist
    if !isdir(fig_save_dir)
        mkdir(fig_save_dir)
    end

    for idx_u in 1:n_params
        # create the density plot
        p=StatsPlots.plot(chain_params.:value.:data[:,idx_u,:][:],seriestype=:density,label="Density",linewidth=linewidth,legend=:outertop,legend_columns=3,legendfontsize=my_font_size,tickfontsize=my_font_size,right_margin=right_margin)
        
        # vertical line for lower hpd interval
        vline!(lower_hpd[idx_u:idx_u],label="Lower HPD",linewidth=linewidth,linestyle=:dashdot)
        
        # vertical line for upper hpd interval
        vline!(upper_hpd[idx_u:idx_u],label="Upper HPD",linewidth=linewidth,linestyle=:dashdot)
        
        
        # save the plot
        savefig(p,fig_save_dir*fig_save_name*"-"*names_all[idx_u]*"-density.pdf")

        # trace plots
        p_trace=plot(chain_params.:value.:data[:,idx_u,:][:],label="Trace",linewidth=linewidth,legend=:outertop,legend_columns=3,legendfontsize=my_font_size,tickfontsize=my_font_size)
        
        # horizontal line for lower hpd interval
        hline!(lower_hpd[idx_u:idx_u],label="Lower HPD",linewidth=linewidth,linestyle=:dashdot)
                
        # horizontal line for upper hpd interval
        hline!(upper_hpd[idx_u:idx_u],label="Upper HPD",linewidth=linewidth,linestyle=:dashdot)

        # save the plot
        savefig(p_trace,fig_save_dir*fig_save_name*"-"*names_all[idx_u]*"-trace.pdf")

    end

    return nothing
end