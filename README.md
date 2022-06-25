# Soc-into-to-Machine-Intelligence

This is the overview of the material taught upto week 4 for the Soc Intro to Machine Intelligence
We'll walk through with all the material that has been taught weekwise.

#### Week 1

1. Initally this was our introduction phase, the pace here was slow but steady. Here we were taught about Intsallation of python and getting started with Jupyter notebook.
2. Our second obejctive of this week was to get familiar with interface of Jupyter notebook.

#### Week 2
This week we learnt about :-
##### NumPy Operation
Basics of NumPy:
1. Creating an array
2. Array operations

Multi-dimensional Arrays:
1. Slicing

NumPy libraries

##### Git and Github Fundamentals
Here we are told a bit about things such as master branch, commit, issue etc. 
##### Intro to Neural Networks
 
These are a series of lectures which tell about the learning and help in understanding neural networks.

Initially we were taught about what neural networks are via a house price prediction example. Here we learnt that the basic structure of a neural network is input----> neuron-----> output.
As complextity increase (as in the parameters to  find an outcome) the structure becomes more dense and includes some more layer which we refer to as hidden layers.

<img src = "https://otexts.com/fpp2/nnet2.png">

Here we were also told about ReLU functions, ReLU standing for REctified Linear Unit are the functions used by neurons to take the input and give the output.

Next we were introduce dto the concept of Binary Classification ==> output {0,1}.
Also some notations were introduced in these lectures.

From here on we went onto learning more about NN equations for hidden layer and also the output layer, here we learnt about sigmoid function, loss function etc and then we learnt about NN representation.
From here we move onto computation of NN and here we talked about vectorizing.

Later on we learn about activation functions such as sigmoid function,tan(h) function, sometimes its better to have tan(h) function as we sometimes want our data to be centered around 0, Whereas for binary classifcation it is advisable to use sigmoid function as our y_hat belongs to {0,1}. Though if any of these is not working, best to use would be ReLU functions.Also we need to use Non-linear Activation Functions for some intersting results.
Next we learn about Gradient Descent, for making this a bit familiar we started with derivatives fo Activation Functions. After that we saw formulas for computing derivatives for Activation functions. This provided basis for methods that were taught in the later namely, backward propogation. Generally we go from input to output but backward propogation allows us to know the input from the given output so basically we go from right to left (output---> neuron--->input).

### Week 3-4

This week we were provided videos in continuation of NN. Starting with the the idea of Random initallisation, here we found out why is it neccesary to assign random weights to a NN and what would happen if we initialize weights become 0( error in gradient descent occurs which results it giving same values in every layer).
Now we shift from shallow NN to Deep NN.Starting with the notations for the deep L layered NN and then we procceed to Forward Propogation in Deep network.I'll upload an image that gives the summary of the forward and backward propogation for deep NN.
Then we move onto learn about parameters and Hyperparameters(alpha,hidden layers, hidden units etc). Basically hyperparameters control the parameters in the NN algorithm.
Then we learn about methods such as  Dropout regularization and Early stopping, tecchniques which reduces overfitting in NN. One of the methods that can speed up the training is normalising inputs.Some more problems that are caused in trainig are because of the gradient, someitimes it becomes too small or too big. Vansihing/ exploding gradient helps to overcome this problem.




##### Convolution Neural Network(CNN)

Generally CNN is used to identify images,their recognistion and processing.Basically each image has an iedntifying feature through which we can determine it. The identifying feature is then marked on a Feature Map,the identfyong feature on the feature map is a number which is 1 or close to one. Filters are feature detectors which basically identify the feature activation points on the feature map. To reduce te number of computations done in Deep NN we use the method of pooling. For eg Max pooling is chosing the maximum number in a regoin selected in a feature map, Average pooling is giving the average number of a smaller region specified in a feature map.
An image explaining how CNN has advantage over ANN and how pooling helps in reducing computation load on a computer has been uploaded.

##### Recurrent Neural Network(RNN)
Generally RNN is used for data that requires a sequencially ordered output.Here the output of the previous step acts as input to current step.RNN is like storing memory as in it remembers data from the previous knowledge given to it and uses it when it is required further in the future.
Sometimes RNN and CNN are used together to increase the effectivity in pixels.

### Week 5

In this week we learn about topics namely:-
1. Reinforcement Learning
2. Q-Learning (intro + implementation)

##### Reinforcement Learning

RL is a part of ML, here agents are sel-trained on reward and punishment based mechanisms. Here the best possible action or path is achieved to gain maximum rewards and minimum punishments through observation.
There are various defintions which give an insight to what RL is, I'll provide the one which was most understandable to me:
RL is based on rewarding desired behaviours and punishing undesired ones. Instead of one input producing one output, the algorithm produces various outputs and is trained to select the right one based on some given variables.

Now to familiarise RL topics, some terminologies were introduced

1. Agent:It is the sole decision maker and learner
2. Environment:A physical world where the agent learns and decides the action to be performed.
3. Action: A list of action which an agent can perform
4. State: The current situation of the agent in the nevironment
5. Reward: For each selected action by the agent, the environment gives a reward. It’s usually a scalar value and
nothing but feedback from the environment
6. Policy: The agent prepares strategy(decision-making) to map situations to actions.
7. Value Function – The value of state shows up the reward achieved starting from the state until the policy
is executed
8. Model – Every RL agent doesn’t use a model of its environment. The agent’s view maps state-action pairs
probability distributions over the states

RL Workflow:
Create an Environment --> Define the reward--> Create the agent--> Train and validate the agent--> Deploy the policy.

Reinfoecement Learning Algorithms
There are 3 approacehes to implement reinforcement learning algorithms
1. Value based :The main goal of this method is to maximize a value function. Here, an agent through a policy expects a long-term return of the current states.
2. Policy based: In policy-based, you enable to come up with a strategy that helps to gain maximum rewards in the future through possible actions performed in each state. Two types of policy-based methods are deterministic and stochastic.
3. Model based: In this method, we need to create a virtual model for the agent to help in learning to perform in each specific environment.

##### Types of Reinforcement Learning:
1. Positive Reinforcement:makes a positive impact on behaviour.
2. Negative Reinforcement: when a negative condition is avoided it tries to stop the acion in future cases.

##### Some models used for RL.
1. Markov Decision Process(MDP's)
2. Q learning

We'll moreover look on Q Learning.
'Q' in Q-learning stands for quality as in how useful a given action is.

Q*(s,a) is the expected value of doing a in state s and then following the optimal policy.It uses Temporal Differences(TD) to estimate the value of Q*(s,a). It is an agent learning from an environment.
Agent maintains a table of Q[S,A] where S is the set of states and A is the set of actions.
Note: Q[s,a] represents its current estimate of Q*(s,a).

Q-table:it is a table that will guide us to the best action at each state.
Q-function:it uses the Bellman eqaution and takes two inputs -state(s) and action(a).

Steps for Q-learning Algorithm Process:
1. Initialise the Q-table: A table with n coloumns(n=number of actions) and m rows(m= number of states)
2. Chose an action
3. Perform an action:First, an action (a) in the state (s) is chosen based on the Q-Table. Note that, as mentioned earlier, when the episode initially starts, every Q-value should be 0.
Then, update the Q-values for being at the start and moving right using the Bellman equation which is stated above.
4. Measure reward
5. Evaluate:Q(s,a) function needs to be updated. The process is repeated until the learning is stopped.
Initially, we exlpore the environment and upadte the Q-table.When it is ready, the agent starts to exploit the environment and start taking better actions.

Next we learn about Q-learning implementation in python.
