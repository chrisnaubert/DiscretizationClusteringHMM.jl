#!/usr/bin/env bash
#export PATH="$PATH:/home/chris/.juliaup/bin/"


# estimate the transition matrix
julia -t auto main_hmm.jl \
--data_file ./data/macro-uncty-qtrly.csv \
--idx_data 2 \
--date_start 7/1960 \
--date_end 10/2019 \
--n_samples 10000 \
--my_cluster_method kmedoids_abs \
--n_clusters_max 6 \
--save_sample_dir ./sample-results/ \
--save_sample_name hmm-macro-uncty-kmedoids-abs \

