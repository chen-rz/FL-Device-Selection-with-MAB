# Parameters

# Number of dataset partions (= number of total clients)
pool_size = 100

# Number of participants in each round
num_to_choose = 20

# Global iterations
num_rounds = 10 # TODO Global iter

# Beta
beta = 0.5

# Channel gain
sys_channelGain = 4e-11

# Background noise
sys_bgNoise = 1e-13

# Total bandwidth
sys_bandwidth = 1e6

# Model size (bit) # Obsoleted
# sys_modelSize = 1984192 # LeNet
# sys_modelSize = 1825433920 # AlexNet

# Model flops
sys_modelFlops = 523878 # LeNet
# sys_modelFlops = 712651338 # AlexNet

# Global time constraint (s)
timeConstrGlobal = 2000
