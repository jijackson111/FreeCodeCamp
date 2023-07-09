import tensorflow_probability as tfp
import tensorflow as tf

# Distributions
tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# Create the model
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

# Predict
mean = model.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())