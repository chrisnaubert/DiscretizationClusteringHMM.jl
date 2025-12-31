# DiscretizationClusteringHMM.jl

This package hosts the code for the project "Discretization using Clustering and Hidden Markov Models". Please see these [slides](https://drive.google.com/file/d/1L5SwP3oRLH8F2HiX5f0ljXpmLozvTfpn/view?usp=drive_link) for details on the methodology.

The code uses a two step procedure to discretize a continuous valued stochastic process. In the first step, k-means of k-medoids clustering is used to select the number of hidden states present in the data based on silhouette scores. In the second step, given the number of hidden states, the codes performs Bayesian estimation of the hidden Markov model parameters using Hamiltonian Monte Carlo. One can then use the estimated transition probabilities and state means in a structural dynamic stochastic general equilibrium model.

## User Inputs

The user inputs are specified in ``run_sampling.sh``

- data_file: csv file containing the data. the code assumes that the column names are in the first row and the firs column corresponds to the dates.
- idx_data: column index of the data
- date_start: string specifying the first date of the data file you want to use
- date_end: string specifying the last date of the data file you want to use
- n_samples: the number of samples you want to draw from the posterior
- my_cluster_method: string indicating whether to use kmeans, kmedoids_abs (k-medoids using the absolute value metric) or kmedoids_sq (k-medoids using the euclidian metric)
- n_clusters_max: maximum number of clusters. the optimal number of clusters will be based on the silhouette scores of the clustering results with k=2 to k=n_clusters_max clusters
- save_sample_dir: directory where the results will be saved
- save_sample_name: file name for the saved results

## Posterior Analysis

The files ``my_plotting.jl`` and ``my_tables.jl`` contain code for analyzing the sampling results. This includes creating trace and density plots for the HMM parameters and tables containing the posterior means, medians, high-density intervals, effective sample size percentage and R-hat. See below for example code. 

```
chain_load_name=pwd()*"/sample-results/"*fig_save_name*".jld2"
fig_save_dir=pwd()*"/figures/"
table_save_dir=pwd()*"/tables/"
table_save_name=fig_save_name

# length of the simulation
t_sim=1000

# create the scatter plots and posterior probability plots
plot_scatter_and_probs(chain_load_name,fig_save_dir,fig_save_name)

# create a table with the posterior results
create_posterior_table(chain_load_name,table_save_dir,table_save_name)

# using the posterior estimates, simulate the process and plot the time series
plot_simulated_series(chain_load_name,fig_save_dir,fig_save_name,t_sim)

# create the trace and density plots
plot_trace_and_density(chain_load_name,fig_save_dir*fig_save_name*"/",fig_save_name)
```


## Directory Structure

```
DiscretizationClusteringHMM.jl/
├── data/
│   ├── gpr-qtrly.csv
│   └── macro-uncty-qtrly.csv
├── env-hmm-jl.1.10/
│   ├── Manifest.toml
│   └── Project.toml
├── figures/
│   └── placeholder.txt
├── sample-results/
│   └── placeholder.txt
├── src/
│   ├── base_hmm.jl
|   ├── my_plotting.jl
|   ├── my_tables.jl
│   └── utils.jl
├── tables/
│   ├── placeholder.txt
├── main_hmm.jl
├── run_sampling.sh
└── readme.md
```
