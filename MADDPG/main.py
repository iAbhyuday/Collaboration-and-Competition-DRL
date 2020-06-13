from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb

from MADDPG.hyperparameters import *


def train(maddpg, env, n_episodes=1000, save_every=50):
    """Training loop helper for running the environment using the MADDPG algorithm.
    Params
    ======
        maddpg (MADDPG): instance of MADDPG wrapper class
        env (UnityEnvironment): instance of Unity environment for training
        n_episodes (int): number of episodes to train for
        save_every (int): frequency to save model weights
    """
    widget = [
        "Episode: ", pb.Counter(), '/' , str(n_episodes), ' ', 
        pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ', 
        'Rolling Average: ', pb.FormatLabel('')
    ]
    timer = pb.ProgressBar(widgets=widget, maxval=n_episodes).start()

    solved = False
    scores_total = []
    scores_deque = deque(maxlen=100)
    rolling_score_averages = []
    last_best_score = 0.0

    # Environment information
    brain_name = env.brain_names[0]

    for i_episode in range(1, n_episodes+1):
        current_average = 0.0 if i_episode == 1 else rolling_score_averages[-1]
        widget[12] = pb.FormatLabel(str(current_average)[:6])
        timer.update(i_episode)

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations[:, -STATE_SIZE:]
        scores = np.zeros(NUM_AGENTS)
        maddpg.reset()

        while True:
            actions = maddpg.act(states)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations[:, -STATE_SIZE:]
            rewards = env_info.rewards
            dones = env_info.local_done
            
            maddpg.step(states, actions, rewards, next_states, dones)

            scores += rewards
            states = next_states

            if np.any(dones):
                break

        max_episode_score = np.max(scores)

        scores_deque.append(max_episode_score)
        scores_total.append(max_episode_score)

        average_score = np.mean(scores_deque)
        rolling_score_averages.append(average_score)

        if average_score >= 0.5 and not solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, average_score
            ))
            solved = True
            maddpg.save_model()
            last_best_score = average_score

        if i_episode % save_every == 0 and solved:
            # Only save these weights if they are better than the ones previously saved
            if average_score > last_best_score:
                last_best_score = average_score
                maddpg.save_model()

    return scores_total, rolling_score_averages


def plot_results(scores, rolling_score_averages):
    """Plots training results from the training loop in the `train` method.
    Params
    ======
        scores (list): list of the max among all agents in a given episode
        rolling_score_averages (list): average of max agent scores in the last 100 episodes
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(1, len(scores) + 1), scores, label="Max Score")
    plt.plot(np.arange(1, len(rolling_score_averages) + 1), rolling_score_averages, label="Rolling Average")
    # This line indicates the score at which the environment is considered solved
    plt.axhline(y=0.5, color="r", linestyle="-", label="Environment Solved")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()
    

def test(maddpg, env, num_games=11, load_model=False):
    """Tests the training results by having both agents play a match.
    Params
    ======
        maddpg (MADDPG): instance of MADDPG wrapper class
        env (UnityEnvironment): instance of Unity environment for testing
        num_games (int): number of games to be played
        load_model (bool): choice of loading model weights from their saved location
    """
    if load_model:
        maddpg.load_model()

    print("Agent #0: Red racket")
    print("Agent #1: Blue racket")
    print("---------------------")

    game_scores = [0 for _ in range(NUM_AGENTS)]

    # Environment information
    brain_name = env.brain_names[0]

    for i in range(1, num_games+1):
        env_info = env.reset(train_mode=False)[brain_name]   
        states = env_info.vector_observations
        scores = np.zeros(NUM_AGENTS)

        while True:
            actions = maddpg.act(states)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            scores += rewards
            dones = env_info.local_done

            if np.any(dones):
                winner = np.argmax(scores)
                game_scores[winner] += 1
                print("Partial game score: {}".format(game_scores))
                break

            states = next_states

    print("---------------------")
    print("Winner: Agent #{}".format(np.argmax(game_scores)))
