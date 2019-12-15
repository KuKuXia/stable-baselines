import warnings

import gym
from gym import spaces
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan


def check_env(env, warn=True):
    """
    Check that an environment follows Gym API.
    This is particularly useful when using a custom environment.
    Please take a look at https://github.com/openai/gym/blob/master/gym/core.py
    for more information about the API.

    :param env: (gym.Env) The Gym environment that will be checked
    :param warn: (bool) Whether to output additional warnings
        mainly related to the interaction with Stable Baselines
    """
    # Helper to link to the code, because gym has no proper documentation
    gym_env = " cf https://github.com/openai/gym/blob/master/gym/core.py"
    gym_spaces = " cf https://github.com/openai/gym/blob/master/gym/spaces/"

    assert isinstance(env, gym.Env), "You environment must inherit from gym.Env class" + gym_env
    is_goal_env = isinstance(env, gym.GoalEnv)

    # ============= Check the spaces ================

    assert hasattr(env, 'observation_space'), "You must specify an observation space (cf gym.spaces)" + gym_spaces
    assert hasattr(env, 'action_space'), "You must specify an action space (cf gym.spaces)" + gym_spaces

    observation_space = env.observation_space
    action_space = env.action_space
    # Whether to check that the returned observation is a numpy array
    # it is not mandatory for `Dict` and `Tuple` spaces
    enforce_array_obs = not isinstance(observation_space, (spaces.Dict, spaces.Tuple))

    assert isinstance(observation_space, spaces.Space), "The observation space must inherit from gym.spaces" + gym_spaces
    assert isinstance(action_space, spaces.Space), "The action space must inherit from gym.spaces" + gym_spaces


    if warn and isinstance(observation_space, spaces.Dict) and not is_goal_env:
        warnings.warn("The observation space is a Dict but the environment is not a gym.GoalEnv ({}), "
                      "this is currently not supported by Stable Baselines "
                      "(cf https://github.com/hill-a/stable-baselines/issues/133), you will need to use a custom policy. ".format(gym_env)
                    )

    if warn and isinstance(observation_space, spaces.Tuple):
        warnings.warn("The observation space is a Tuple,"
                      "this is currently not supported by Stable Baselines "
                      "(cf https://github.com/hill-a/stable-baselines/issues/133), "
                      "you will need to flatten the observation and maybe use a custom policy. "
                    )

    # If image, check the low and high values, the type and the number of channels
    # and the shape (minimal value)
    if warn and isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:

        if observation_space.dtype != np.uint8:
            warnings.warn("It seems that your observation is an image but the `dtype` "
                          "of your observation_space is not `np.uint8`. "
                          "If your observation is not an image, we recommend you to flatten the observation "
                          "to have only a 1D vector")

        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            warnings.warn("It seems that your observation space is an image but the "
                          "high and lower bounds are not in [0, 255]. "
                          "Because the CNN policy normalize automatically the observation "
                          "you may encounter issue if the values are not in that range."
                        )
        # Should be check n_channels?
        if observation_space.shape[0] < 36 or observation_space.shape[1] < 36:
            warnings.warn("The minimal resolution for an image is 36x36 for the default CNNPolicy. "
                          "You might need to use a custom `cnn_extractor` "
                          "cf https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html")

    if warn and isinstance(observation_space, spaces.Box) and len(observation_space.shape) not in [1, 3]:
        warnings.warn("Your observation has an unconventional shape (neither an image, nor a 1D vector). "
                      "We recommend you to flatten the observation "
                      "to have only a 1D vector")

    if warn and isinstance(action_space, spaces.Box):
        # if the if-block is not separated `action_space.low` will be evaluated even though
        # `isinstance(action_space, spaces.Box)` is False...
        if np.abs(action_space.low) != np.abs(action_space.high) or np.abs(action_space.low) > 1 or np.abs(action_space.high) > 1:
            warnings.warn("We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) "
                          "cf https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html")


    # ============ Check the returned values ===============

    # because env inherits from gym.Env, we assume that `reset()` and `step()` methods exists
    obs = env.reset()

    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(obs, tuple), "The `reset()` method should only return one value, not a tuple"

    # The check for a GoalEnv is done by the base class
    if not is_goal_env:
        if isinstance(observation_space, spaces.Discrete):
            assert isinstance(obs, int), "The observation returned by `step()` method must be an int"
        elif enforce_array_obs:
            assert isinstance(obs, np.ndarray), "The observation returned by `reset()` method must be a numpy array"

    assert observation_space.contains(obs), "The observation returned by the `reset()` method does not match the given observation space"

    # Sample a random action
    action = action_space.sample()
    data = env.step(action)

    assert len(data) == 4, "The `step()` method must return four values: obs, reward, done, info"

    # Unpack
    obs, reward, done, info = data

    if not is_goal_env:
        if isinstance(observation_space, spaces.Discrete):
            assert isinstance(obs, int), "The observation returned by `step()` method must be an int"
        elif enforce_array_obs:
            assert isinstance(obs, np.ndarray), "The observation returned by `step()` method must be a numpy array"
    assert observation_space.contains(obs), "The observation returned by the `step()` method does not match the given observation space"

    # We also allow int because the reward will be cast to float
    assert isinstance(reward, (float, int)), "The reward returned by `step()` must be a float"
    assert isinstance(done, bool) or float(done) in (0, 1), "The `done` signal must be a boolean"
    assert isinstance(info, dict), "The `info` returned by `step()` must be a python dictionary"

    if isinstance(env, gym.GoalEnv):
        # For a GoalEnv, the keys are check at reset
        assert reward == env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

    # TODO: should we check metadata['render.modes'] ?

    # Check for Inf and NaN using the VecWrapper
    vec_env = VecCheckNan(DummyVecEnv([lambda: env]))

    # The check only works with numpy arrays
    if enforce_array_obs:
        for _ in range(10):
            action = [action_space.sample()]
            _, _, _, _ = vec_env.step(action)
