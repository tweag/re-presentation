import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp

tfd = tfp.distributions

which_mixture = '1d'
# which_mixture = '2d'

if which_mixture == '1d':
    means = [-1.5, 2.0]
    stds = [0.5, 0.2]
    weights = np.array([0.3, 0.7])
    weights = weights / weights.sum()
    categorial = tfd.Categorical(probs=weights.astype(np.float32))
    mixture = tfd.Mixture(cat=categorial,
                          components=[tfd.Normal(loc=mean, scale=std)
                                      for mean, std in zip(means, stds)])

if which_mixture == '2d':
    weights = np.ones(9) #np.random.uniform(size=9)
    sl = int(np.sqrt(len(weights)))
    means = np.array([[3 * (i % sl), 3 * (i // sl)]
                      for i in range(len(weights))], dtype=np.float32)

    weights = weights / weights.sum()
    stds = np.array([np.ones(2) * 0.25] * len(weights), dtype=np.float32
    )
    categorial = tfd.Categorical(probs=weights.astype(np.float32))
    mixture = tfd.Mixture(
        cat=categorial,
        components=[tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
                    for mean, std in zip(means, stds)]
    )

# inverse_temperatures = 0.15 ** tf.range(16, dtype=np.float32)
inverse_temperatures = np.array([1.0, 0.8, 0.6, 0.4, 0.1], dtype=np.float32)

stepsizes = 0.7 ** inverse_temperatures
if which_mixture == '2d':
    stepsizes = np.array(stepsizes)[:,None].repeat(mixture.event_shape[0], axis=1)

# we start the Markov chain with some random initial state
initial_state = np.random.uniform(
    low=-5, high=5, size=mixture.event_shape
).astype(np.float32)

def make_kernel_fn(target_log_prob_fn):
    # we want to do local sampling with the Metropolis algorithm
    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_uniform_fn(stepsizes)
    )
    return kernel


def trace_swaps(unused_state, results):
    return (results.is_swap_proposed_adjacent,
            results.is_swap_accepted_adjacent,
            results.post_swap_replica_results.is_accepted,
    )

# decorating that function with tf.function builds a static compute graph,
# which is much faster
@tf.function(experimental_compile=True)
def run_chain(initial_state, num_results=1000):
    remc_kernel = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=mixture.log_prob,
        inverse_temperatures=inverse_temperatures,
        # swap_proposal_fn=tfp.mcmc.default_swap_proposal_fn(1.0),
        swap_proposal_fn=tfp.mcmc.even_odd_swap_proposal_fn(1. / 5000000000.),
        # swap_proposal_fn=tfp.mcmc.even_odd_swap_proposal_fn(1. / 5.),
        make_kernel_fn=make_kernel_fn)

    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=initial_state,
        kernel=remc_kernel,
        num_burnin_steps=500,
        trace_fn=trace_swaps)


n_samples = 10000

samples, (swap_proposed, swap_accepted, hmc_accepted) = (
    run_chain(initial_state, n_samples)
)

samples = samples.numpy()
try:
    inverse_temperatures = inverse_temperatures.numpy()
except:
    pass


print("")
print("Inverse temperatures:", inverse_temperatures)
p_accs = swap_accepted.numpy().sum(0) / swap_proposed.numpy().sum(0)
print("RE acceptance rates:", *["{:.2f}".format(p) for p in p_accs])
local_p_accs = hmc_accepted.numpy().mean(0)
print("RWMC acceptance rates:", *["{:.2f}".format(p) for p in local_p_accs])


# Detect swaps. This method works only under the assumption that
# when performing local MCMC moves, starting from two different 
# initial states, you cannot end up with the same state
swaps = {}
for i in range(n_samples):
    matches = np.where(swap_accepted.numpy()[i])[0]
    if len(matches) > 0:
        swaps[i] = matches


# Reconstruct trajectories of single states through the temperature
# ladder
def reconstruct_trajectory(start_index, n_chains):
    res = []
    current_ens = start_index
    for i in range(n_samples):
        res.append(current_ens)
        if i in swaps:
            if current_ens in swaps[i]:
                current_ens += 1
            elif current_ens in swaps[i] + 1:
                current_ens -= 1

    return np.array(res)


def plot_state_trajectories(trajectories, ax, max_samples=300):
    for trajectory in trajectories:
        ax.plot(trajectory[:max_samples], lw=2)
    ax.set_xlabel("# of MCMC samples")
    ax.set_ylabel(r"inverse temperature $\beta$")
    ax.set_yticks(range(len(inverse_temperatures)))
    ax.set_yticklabels(inverse_temperatures)
    

# which states to follow
# start_state_indices = np.arange(len(inverse_temperatures))[::3]
start_state_indices = (0,)

fig, ax = plt.subplots(figsize=(8, 6))
trajectories = np.array([reconstruct_trajectory(i, len(inverse_temperatures)) 
                         for i in start_state_indices])
plot_state_trajectories(trajectories, ax, max_samples=1000)
fig.tight_layout()
plt.show()

if which_mixture == '1d':
    fig, ax = plt.subplots()
    ax.hist(samples, bins=50, density=True)
    xspace = np.linspace(-4, 5, 1000)
    ax.plot(xspace, np.exp(mixture.log_prob(xspace).numpy()))
    for spine in ('left', 'top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.set_yticks(())
    plt.show()

if which_mixture == '2d':
    fig, (ax1, ax2) = plt.subplots(1,2)
    space = np.linspace(-2, 8, 100)
    ax1.hist2d(*samples.T, bins=space, density=True)
    ax1.set_aspect('equal')
    ax1.set_xticks(())
    ax1.set_yticks(())
    mg = np.dstack(np.meshgrid(space, space))
    lps = mixture.log_prob(mg)
    ax2.matshow(lps.numpy(), origin='lower')
    ax2.set_xticks(())
    ax2.set_yticks(())
    plt.show()
