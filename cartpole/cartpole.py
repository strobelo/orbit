#!/s/bach/j/under/dstrobel/miniconda3/bin/python
import argparse
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import wrappers
import time

from agents import QAgent



class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    env.seed(0)
    #gammaF = lambda gp: 1 - 0.98*(1-gp)
    gammaF = lambda gp: 0.7
    agent = QAgent(env.action_space, env.observation_space, use_cuda=True, qlr=0.15, qLayers=6, qhidden=32, gammaF=gammaF,eld=0.99993,replaySize=10000)

    episode_count = 20000
    reward = 0
    done = False

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/qlr-results'
    #env = wrappers.Monitor(env, directory=outdir, force=False)
    allRewards = []
    allActions = []
    observations, rewards, actions = [],[],[]
    oldo, oldr, olda = [],[],[]
    startTime = time.time()
    printSteps = list(np.linspace(0,episode_count,episode_count/50,dtype=int))
    p = printSteps.pop(0)
    try:
        for i in range(episode_count):
            ob = env.reset()
            while True:
                action = agent.act(ob, reward)
                
                ob, reward, done, _ = env.step(action)
                observations.append(ob)
                rewards.append(reward)
                actions.append(action)
                if done:
                    break
            allActions += (actions)
            allRewards.append(sum(rewards))
            agent.trainQ(observations,actions,rewards,32)
            if not agent.replay.isEmpty():
                agent.trainQ(*agent.replay.sample(min(i,1024)),24, decay=False)
            oldo, olda, oldr = observations, actions, rewards
            observations, rewards, actions = [],[],[]
            if i == p:
                print('Episode {} complete; {:2f}s elapsed; epsilon={:3f}; Scored {}'.format(i,time.time()-startTime, agent.epsilon, allRewards[i]))
                p = printSteps.pop(0)


                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.
    except KeyboardInterrupt:
        pass
    # Close the env and write monitor result info to disk
    r = np.array(allRewards)
    print('Average Reward over {} episodes: {}'.format(episode_count,np.mean(r)))
    fig,ax=plt.subplots(1,3)
    ax[0].plot(r)
    ax[0].set_title('Reward Achieved')
    ax[1].plot(agent.lossTrace)
    ax[1].set_title('Loss Trace')
    ax[2].plot(allActions,'.',alpha=0.2)
    ax[2].set_title('Actions taken')
    fig.show()
    plt.show()
    env.close()



