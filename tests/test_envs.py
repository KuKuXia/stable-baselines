import pytest
import gym
from gym import spaces
import numpy as np

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox


@pytest.mark.parametrize("env_id", ['CartPole-v0', 'Pendulum-v0', 'BreakoutNoFrameskip-v4'])
def test_env(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    env = gym.make(env_id)
    with pytest.warns(None) as record:
        check_env(env)

    # Pendulum-v0 will produce a warning because the action space is
    # in [-2, 2] and not [-1, 1]
    if env_id == 'Pendulum-v0':
        assert len(record) == 1
    else:
        # The other environments must pass without warning
        assert len(record) == 0


@pytest.mark.parametrize("env_class", [IdentityEnv, IdentityEnvBox, BitFlippingEnv])
def test_custom_envs(env_class):
    env = env_class()
    check_env(env)


def test_failures_and_warnings():
    """
    Test that common failure cases are caught
    """
    env = gym.make('BreakoutNoFrameskip-v4')
    # Change the observation space
    non_default_spaces = [
        # Small image
        spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
        # Range not in [0, 255]
        spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.uint8),
        # Wrong dtype
        spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.float32),
        # Not an image, it should be a 1D vector
        spaces.Box(low=-1, high=1, shape=(64, 3), dtype=np.float32),
        # Tuple space is not supported by SB
        spaces.Tuple([spaces.Discrete(5), spaces.Discrete(10)]),
        # Dict space is not supported by SB when env is not a GoalEnv
        spaces.Dict({"position": spaces.Discrete(5)}),
    ]
    for new_obs_space in non_default_spaces:
        env.observation_space = new_obs_space
        # Patch methods to avoid errors
        env.reset = new_obs_space.sample
        def patched_step(_action):
            return new_obs_space.sample(), 0.0, False, {}
        env.step = patched_step
        with pytest.warns(UserWarning):
            check_env(env)

    env = IdentityEnvBox()
    # Return an observation that does not match the observation_space
    def reset_wrong_shape():
        return np.ones((3,))

    env.reset = reset_wrong_shape
    with pytest.raises(AssertionError):
        check_env(env)

    # Return not only the observation
    def reset_tuple():
        return env.observation_space.sample(), False

    env.reset = reset_tuple
    with pytest.raises(AssertionError):
        check_env(env)

    # Return a wrong reward
    def step_wrong_reward(_action):
        return env.observation_space.sample(), np.ones(1), False, {}

    env.step = step_wrong_reward
    with pytest.raises(AssertionError):
        check_env(env)

    # Info dict is not returned
    def step_no_info(_action):
        return env.observation_space.sample(), 0.0, False

    env.step = step_no_info
    with pytest.raises(AssertionError):
        check_env(env)

    # Done is not a boolean
    def step_wrong_done(_action):
        return env.observation_space.sample(), 0.0, 3.0, False

    env.step = step_wrong_done
    with pytest.raises(AssertionError):
        check_env(env)
