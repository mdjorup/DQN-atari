import numpy as np
import tensorflow as tf

from preprocessing import phi

import replay_memory





memory = replay_memory.ReplayMemory(10000)


print(memory.curr_capacity)
# macros


