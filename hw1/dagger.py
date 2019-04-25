import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


def build_model(training_data, env, config):
    from keras.models import Sequential
    from keras.layers import Dense, Lambda
    from keras.optimizers import Adam
    from sklearn.utils import shuffle

    obs_mean, obs_std = np.mean(training_data['observations']), np.std(
        training_data['observations'])

    obs_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]

    model = Sequential([
        Lambda(lambda obs: (obs - obs_mean) / obs_std),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(actions_dim)
    ])

    opt = Adam(lr=config['learning_rate'])
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])

    x, y = shuffle(training_data['observations'], training_data['actions'].reshape(
        [-1, actions_dim]))  # because validation data is extracted before shuffling
    model.fit(x, y, batch_size=128, validation_split=0.1,
              epochs=config['epochs'], verbose=2)
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    args = parser.parse_args()

    training_data = {}
    print('loading training data')
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        training_data = pickle.load(f)
    print('loaded and start to train policy')

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    config = {
        'learning_rate': 1e-3,
        'num_rollouts': 30,
        'epochs': args.epochs
    }

    expert_policy = load_policy.load_policy(args.expert_policy_file)
    
    with tf.Session():
        tf_util.initialize()
        
        returns = []
        observations = []
        actions = []
        for i in range(config['num_rollouts']):
            policy_fn = build_model(training_data, env, config)
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn.predict(obs[None, :])
                observations.append(obs)
                actions.append(expert_policy(obs[None,:]))
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
            
            training_data['observations'] = np.concatenate([training_data['observations'], np.array(observations)])
            training_data['actions'] = np.concatenate([training_data['actions'], np.array(actions)])


        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
