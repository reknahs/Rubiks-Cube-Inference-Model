import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from numpy.core.fromnumeric import mean
from stable_baselines3.common.callbacks import BaseCallback
import os
import optuna
import torch

class Cube:
    def __init__(self, cube):
        self.colors = np.array(cube)

    def D(self):
        state = self.colors
        newF = state[42], state[43], state[44]
        newR = state[51], state[52], state[53]
        newB = state[6], state[7], state[8]
        newL = state[24], state[25], state[26]
        # down is the 4th, so 27-36
        newD = [state[33], state[30], state[27], state[34], state[31], state[28], state[35], state[32], state[29]]
        state[24:27] = newF
        state[42:45] = newR
        state[51:54] = newB
        state[6:9] = newL
        state[27:36] = newD

    def Dp(self):
        self.D()
        self.D()
        self.D()

    def D2(self):
        self.D()
        self.D()

    def U(self):
        state = self.colors
        newF = state[36], state[37], state[38]
        newR = state[45], state[46], state[47]
        newB = state[0], state[1], state[2]
        newL = state[18], state[19], state[20]
        # up is the 2nd so 9-18
        newU = [state[15], state[12], state[9], state[16], state[13], state[10], state[17], state[14], state[11]]
        state[18:21] = newF
        state[36:39] = newR
        state[45:48] = newB
        state[0:3] = newL
        state[9:18] = newU

    def Up(self):
        self.U()
        self.U()
        self.U()

    def U2(self):
        self.U()
        self.U()

    def F(self):
        state = self.colors
        newU = state[8], state[5], state[2]
        newR = state[15], state[16], state[17]
        newD = state[36], state[39], state[42]
        newL = state[33], state[34], state[35]
        # front is the 3rd so 18-27
        newF = [state[24], state[21], state[18], state[25], state[22], state[19], state[26], state[23], state[20]]
        state[15:18] = newU
        state[36], state[39], state[42] = newR
        state[33:36] = newD
        state[8], state[5], state[2] = newL
        state[18:27] = newF

    def Fp(self):
        self.F()
        self.F()
        self.F()

    def F2(self):
        self.F()
        self.F()

    def B(self):
        state = self.colors
        newU = state[38], state[41], state[44]
        newR = state[27], state[28], state[29]
        newD = state[0], state[3], state[6]
        newL = state[11], state[10], state[9]
        # back is the 6th so 45-54
        newB = [state[51], state[48], state[45], state[52], state[49], state[46], state[53], state[50], state[47]]
        state[9:12] = newU
        state[0], state[3], state[6] = newL
        state[27:30] = newD
        state[38], state[41], state[44] = newR
        state[45:54] = newB
    def Bp(self):
        self.B()
        self.B()
        self.B()

    def B2(self):
        self.B()
        self.B()

    def R(self):
        state = self.colors
        newF = state[33], state[30], state[27]
        newU = state[20], state[23], state[26]
        newB = state[17], state[14], state[11]
        newD = state[51], state[48], state[45]
        # right is the 5th so 36-45
        newR = [state[42], state[39], state[36], state[43], state[40], state[37], state[44], state[41], state[38]]

        state[20], state[23], state[26] = newF
        state[11], state[14], state[17] = newU
        state[45], state[48], state[51], = newB
        state[27], state[30], state[33] = newD
        state[36:45] = newR

    def Rp(self):
        self.R()
        self.R()
        self.R()

    def R2(self):
        self.R()
        self.R()

    def L(self):
        state = self.colors
        newF = state[9], state[12], state[15]
        newU = state[53], state[50], state[47]
        newB = state[29], state[32], state[35]
        newD = state[24], state[21], state[18]
        # left is the 1st so 0-9
        newL = [state[6], state[3], state[0], state[7], state[4], state[1], state[8], state[5], state[2]]

        state[18], state[21], state[24] = newF
        state[9], state[12], state[15] = newU
        state[53], state[50], state[47] = newB
        state[29], state[32], state[35] = newD
        state[0:9] = newL
    def Lp(self):
        self.L()
        self.L()
        self.L()

    def L2(self):
        self.L()
        self.L()

    def solved(self):
        state = self.colors
        if len(set(state[0:9])) == 1:
            if len(set(state[9:18])) == 1:
                if len(set(state[18:27])) == 1:
                    if len(set(state[27:36])) == 1:
                        if len(set(state[36:45])) == 1:
                            if len(set(state[45:54])) == 1:
                                return True
        return False

    def __str__(self):
        return str(list(self.colors))

    def __eq__(self, other):
        return (self.colors == other.colors).all()


class CubeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(6)
        self.observation_space = MultiDiscrete(np.array([6 for i in range(54)]))
        self.state = self.generateScramble()
        self.count = 0

    def step(self, action):

        self.count += 1

        cube = Cube(self.state)

        if action == 0:
            cube.F()
        elif action == 1:
            cube.B()
        elif action == 2:
            cube.R()
        elif action == 3:
            cube.L()
        elif action == 4:
            cube.U()
        elif action == 5:
            cube.D()

        self.state = cube.colors

        side = 0
        for i in range(0, 54, 9):
            if self.state[i+4] == 4:
                side = i

        reward = 0
        if self.state[side+1] == 5:
            reward += 1
        else:
            reward -= 1
        if self.state[side+3] == 5:
            reward += 1
        else:
            reward -= 1
        if self.state[side+5] == 5:
            reward += 1
        else:
            reward -= 1
        if self.state[side+7] == 5:
            reward += 1
        else:
            reward -= 1

        done = reward == 4
        if not done:
          if self.count > 1000:
            done = True
            reward = -500

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = self.generateScramble()
        self.count = 0
        return self.state

    def generateScramble(self):

        solved = [0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1, 1,
                  5, 5, 5, 5, 5, 5, 5, 5, 5,
                  2, 2, 2, 2, 2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4, 4, 4, 4, 4]

        solvedCube = Cube(solved)

        for i in range(random.randint(25,80)):
            move = random.randint(0,5)
            if move == 0:
                solvedCube.F()
            elif move == 1:
                solvedCube.B()
            elif move == 2:
                solvedCube.R()
            elif move == 3:
                solvedCube.L()
            elif move == 4:
                solvedCube.U()
            else:
                solvedCube.D()

        return solvedCube.colors
    def close(self):
        self.state = None


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


def getState(cube):
    state = cube.colors
    pos = []
    for i in range(0, 54, 9):
        if state[i+4] == 4: pos.append(i+4)
    for i in range(1, 9, 2):
        if state[i] == 5: pos.append(i)
    for i in range(10, 18, 2):
        if state[i] == 5: pos.append(i)
    for i in range(19, 27, 2):
        if state[i] == 5: pos.append(i)
    for i in range(28, 36, 2):
        if state[i] == 5: pos.append(i)
    for i in range(37, 45, 2):
        if state[i] == 5: pos.append(i)
    for i in range(46, 54, 2):
        if state[i] == 5: pos.append(i)
    return pos


def simplifyMoves(actions, origin):

    i = 3
    while i < len(actions) and i >= 0:
        if actions[i] == actions[i - 1] and actions[i - 1] == actions[i - 2] and actions[i - 2] == actions[i - 3]:
            actions[i - 3:i + 1] = []
            i -= 4
        i += 1
    c = Cube(origin)
    states = [getState(c)]
    real_states = [c]
    newActions = []
    for i in range(len(actions)):
        move = actions[i]
        lastState = Cube(real_states[-1].colors)
        if move == "F":
            lastState.F()
        elif move == "B":
            lastState.B()
        elif move == "R":
            lastState.R()
        elif move == "L":
            lastState.L()
        elif move == "U":
            lastState.U()
        else:
            lastState.D()

        if getState(lastState) in states:
            real_states[states.index(getState(lastState)):] = []
            newActions[states.index(getState(lastState)):] = []
            states[states.index(getState(lastState)):] = []
        else:
            newActions.append(actions[i])
        real_states.append(lastState)
        states.append(getState(lastState))
    actions = newActions

    i = 2
    while i < len(actions) and i >= 0:
        if actions[i] == actions[i - 1] and actions[i - 1] == actions[i - 2]:
            actions[i] += "'"
            actions[i - 2:i] = []
            i -= 2
        i += 1
    i = 1
    while i < len(actions) and i >= 0:
        if actions[i] == actions[i - 1]:
            actions[i] += "2"
            actions[i - 1:i] = []
            i -= 1
        i += 1


    return actions


def new_evaluate_policy(model, episode_num):
    translation = ["F", "B", "R", "L", "U", "D"]
    episode_lengths = []
    env = CubeEnv()
    for i in range(episode_num):
        obs = env.reset()
        first = obs
        actions = []
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            actions.append(translation[action])
            if done:
                break
        episode_lengths.append(len(simplifyMoves(actions, first)))
    return episode_lengths


LOG_DIR = "./logs/"
OPT_DIR = "./opt/"


def optimize_ppo(trial):
    return {
        "n_steps": trial.suggest_int("n_steps", 2048, 8192),
        "gamma": trial.suggest_loguniform("gamma", 0.8, 0.9999),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-4),
        "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.4),
        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.8, 0.99)
    }


def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)
        env = CubeEnv()
        env = Monitor(env, LOG_DIR)

        rlmodel = PPO("MlpPolicy", env, verbose = 0, **model_params)
        rlmodel.learn(total_timesteps = 50000)
        episode_lengths = new_evaluate_policy(rlmodel, 20)
        real_mean_reward = 500-mean(episode_lengths)
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, "trial{}".format(trial.number))
        rlmodel.save(SAVE_PATH)

        return real_mean_reward
    except Exception as e:
        return -1000


# study = optuna.create_study(direction = "maximize")
# study.optimize(optimize_agent, n_trials = 100, n_jobs = 1)

model = PPO.load(os.path.join(OPT_DIR, "trial95.zip"))

CHECKPOINT_DIR = "./train/"
callback = TrainAndLoggingCallback(check_freq=6, save_path=CHECKPOINT_DIR)

env = CubeEnv()
env = Monitor(env, LOG_DIR)

model_params = {"n_steps": model.n_steps,
                "gamma": model.gamma,
                "learning_rate": model.learning_rate,
                "clip_range": model.clip_range,
                "gae_lambda": model.gae_lambda}

model_params["n_steps"] = 6016

# model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
#
# model.load(os.path.join(OPT_DIR, "trial95.zip"))
#
# model.learn(total_timesteps = 200000, callback = callback)

FINAL_MODEL_PATH = os.path.join("rlmodel_cross.zip")
# model.save(FINAL_MODEL_PATH)

model = PPO.load(FINAL_MODEL_PATH, env = env)
model = PPO.load("./train/best_model_60162.zip")

# print(mean(new_evaluate_policy(model, 100)))









