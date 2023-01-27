import random
import gymnasium
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

"""
Authors: Pawel Dondziak, Jakub Swiderski
this app shows our attempt at making an app that uses keras, keras-rl and numpy for ai that plays a game from Atari called Assault.
https://gymnasium.farama.org/environments/atari/assault/
Unfortunately, we were unable to make the app work. After hours of searching for answers on blogs and forums, we came to the conclusion,
that versions of libraries and python mismatch to one another, but we couldn't match the right versions.
Error we faced: "Exception has occurred: AttributeError 'int' object has no attribute 'shape' when calling dqn.fit()"
if you comment lines from 67 to 72, you can see the game is being displayed, and app making random moves. :)
"""

def build_model(h, w, c, a):
    """
    This function generates out AI model
    :param h: height of shape
    :param w: width of shape
    :param c: channels
    :param a: actions
    :return: model
    """
    m = Sequential()
    m.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(1, h, w, c)))
    m.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    m.add(Convolution2D(64, (3, 3), activation='relu'))
    m.add(Flatten())
    m.add(Dense(512, activation='relu'))
    m.add(Dense(256, activation='relu'))
    m.add(Dense(a, activation='relu'))
    return m


def build_agent(m, a):
    """

    :param m: model
    :param a: actions
    :return: agent
    """
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2,
                                  nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=m, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=a, nb_steps_warmup=1000
                   )
    return dqn


if __name__ == '__main__':
    env = gymnasium.make("ALE/Assault-v5", render_mode="human")
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    # print(actions)
    # print(env.unwrapped.get_action_meanings())

    model = build_model(height, width, channels, actions)
    model.summary()

    game = build_agent(model, actions)
    game.compile(Adam(lr=1e-4))
    game.fit(env, nb_steps=10, visualize=False, verbose=2)

    episodes = 100
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = random.choice([0, 1, 2, 3, 4, 5, 6])
            n_state, reward, truncated, info, done = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))

    env.close()
