from train import make_env,train
from rewards.align import align

env=make_env(4)
reward_mechanism=align(4)
train(env,reward_mechanism)