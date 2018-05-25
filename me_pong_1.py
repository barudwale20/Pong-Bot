import gym
import numpy as np
import pickle

def sampleDown(image):
    return image[::2, ::2, :]

def black_and_white(image):
    return image[:, :, 0]

def clear_bg(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def update_weights(weights, expectationGsquared, g_dict, decay_rate, learning_rate):

    epsilon = 1e-5
    for name_of_layer in weights.keys():
        g = g_dict[name_of_layer]
        expectationGsquared[name_of_layer] = decay_rate * expectationGsquared[name_of_layer] + (1 - decay_rate) * g**2
        weights[name_of_layer] += (learning_rate * g)/(np.sqrt(expectationGsquared[name_of_layer] + epsilon))
        g_dict[name_of_layer] = np.zeros_like(weights[name_of_layer]) # reset batch gradient buffer

        #values of weights being stored

        f = open("file.pkl","w")
        pickle.dump(weights,f)
        f.close()


        # thefile = open('test.txt', 'w')
        # thefile.write(str(weights))
        # thefile.write("\n")
        # thefile.close()
        print (weights)
        # thefile = open('test.txt', 'w')
        # for item in weights:
        #     for x in item:
        #         thefile.write("%s" % x)
        #     thefile.write("\n")
        # thefile.close()



def preprocess(input_observation, prev_processedObs, inputDim):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processedObs = input_observation[35:195] # crop
    processedObs = sampleDown(processedObs)
    processedObs = black_and_white(processedObs)
    processedObs = clear_bg(processedObs)
    processedObs[processedObs != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processedObs = processedObs.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processedObs is not None:
        input_observation = processedObs - prev_processedObs
    else:
        input_observation = np.zeros(inputDim)
    # store the previous frame so we can subtract from it next time
    prevProcessObs = processedObs
    return input_observation, prevProcessObs




def relu(vector):
    vector[vector < 0] = 0
    return vector


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hiddenLayerVals = np.dot(weights['1'], observation_matrix)
    hiddenLayerVals = relu(hiddenLayerVals)
    outputLayerVals = np.dot(hiddenLayerVals, weights['2'])
    outputLayerVals = sigmoid(outputLayerVals)
    return hiddenLayerVals, outputLayerVals



def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        return 2  #up
    else:
        return 3


def discount_for_reward(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discountEpisodeRewards = discountReward(episode_rewards, gamma)
    discountEpisodeRewards -= np.mean(discountEpisodeRewards)
    discountEpisodeRewards /= np.std(discountEpisodeRewards)
    return gradient_log_p * discountEpisodeRewards





def discountReward(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def gradientCalc(gradient_log_p, hiddenLayerVals, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hiddenLayerVals.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def main():
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    # hyperparameters
    episode = 0
    SizeOfBatch = 5
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_neurons = 200
    inputDim = 80 * 80
    learning_rate = 1e-2

    episode = 0
    SumOfReward = 0
    running_reward = None
    prevProcessObs = None

#    if(open( "file.pkl", "r" )):
    weights = pickle.load( open( "data.pkl", "r" ) )
#    else:
    # weights = {
    #         '1': np.random.randn(num_hidden_neurons, inputDim) / np.sqrt(inputDim),
    #         '2': np.random.randn(num_hidden_neurons) / np.sqrt(num_hidden_neurons)
    #         }

    
    expectationGsquared = {}
    g_dict = {}
    for name_of_layer in weights.keys  ():
        expectationGsquared[name_of_layer] = np.zeros_like(weights[name_of_layer])
        g_dict[name_of_layer] = np.zeros_like(weights[name_of_layer])

    episodewiseHiddenLayerValues, episodeObs, episodeLogGradient, episode_rewards = [], [], [], []


    while True:
        env.render()
        processedObs, prevProcessObs = preprocess(observation, prevProcessObs, inputDim)
        hiddenLayerVals, UpShiftProbability = neural_nets(processedObs, weights)

        episodeObs.append(processedObs)
        episodewiseHiddenLayerValues.append(hiddenLayerVals)

        action = choose_action(UpShiftProbability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        SumOfReward += reward
        episode_rewards.append(reward)

        fakeLabels = 1 if action == 2 else 0
        lossFunctionGrad = fakeLabels - UpShiftProbability
        episodeLogGradient.append(lossFunctionGrad)


        if done: # an episode finished
            episode += 1

            # Combine the following values for the episode
            episodewiseHiddenLayerValues = np.vstack(episodewiseHiddenLayerValues)
            episodeObs = np.vstack(episodeObs)
            episodeLogGradient = np.vstack(episodeLogGradient)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episodeLogGradient_discounted = discount_for_reward(episodeLogGradient, episode_rewards, gamma)

            gradient = gradientCalc(
              episodeLogGradient_discounted,
              episodewiseHiddenLayerValues,
              episodeObs,
              weights
            )

            # Sum the gradient for use when we hit the batch size
            for name_of_layer in gradient:
                g_dict[name_of_layer] += gradient[name_of_layer]

            if episode % SizeOfBatch == 0:
                update_weights(weights, expectationGsquared, g_dict, decay_rate, learning_rate)

            episodewiseHiddenLayerValues, episodeObs, episodeLogGradient, episode_rewards = [], [], [], [] # reset values
            observation = env.reset() # reset env
            running_reward = SumOfReward if running_reward is None else running_reward * 0.99 + SumOfReward * 0.01
            print 'resetting env. episode reward total was %f. running mean: %f' % (SumOfReward, running_reward)
            SumOfReward = 0
            prevProcessObs = None

main()
