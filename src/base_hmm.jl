
"""
    bayes_hmm(obs,n_states,hyper_parameter_means,hyper_parameter_stdevs,include_hidden_states)

args:

    obs:
    n_states:
    hyper_parameter_means:
    hyper_parameter_stdevs:
    include_hidden_states:



"""
@model function bayes_hmm(obs,n_states,hyper_parameter_means,hyper_parameter_stdevs,include_hidden_states=false)

    # Emission matrix.
    obs_m = Vector(undef, n_states)
    obs_s = Vector(undef, n_states)

    # Assign distributions to each element
    # of the transition matrix and the
    # emission matrix.
    for idx_s in 1:n_states
        obs_m[idx_s] ~ Normal(hyper_parameter_means[idx_s], hyper_parameter_stdevs[idx_s])
        ig_alpha,ig_theta=inverse_gamma_transform(hyper_parameter_stdevs[idx_s],0.1*hyper_parameter_stdevs[idx_s])
        obs_s[idx_s] ~ InverseGamma(ig_alpha,ig_theta)
    end

    T ~ filldist(Dirichlet(fill(1/n_states, n_states)), n_states)

    hmm = HMM(softmax(ones(n_states)), copy(T'), [Normal(obs_m[i], obs_s[i]) for i in 1:n_states])
    @addlogprob! logdensityof(hmm, obs)

    if include_hidden_states
        seq,_=viterbi(hmm,obs)
        s:=seq
    end

end
