# Self attention is a mechanism to parse the input data 
# Will proceed with simplified self attention -> self-attention -> causal self-attention -> Multi-head attention (adv causal attention)

# Why care about self-attention? RNNs were standard before, which stored the input in latent memory while generating the output. Called memory cell/hidden state
# The memory cell approach lost some information which impacted performance

# Attention allows selective access of the actual input tokens, not just a representation of them, which improves performance by not loosing info

# There are 'importance scores/attention weights' which assign a relative importance to each input token

# Question: what assigns the attention weights? How does that work?

# Note: the alpha values are the attention weights. Before we compute alpha, we compute attention score, omega

import torch 

# ---------------------------------
# the Simplified Self-Attention mechanism
# ---------------------------------

# Step 1: instantiate the input tensor of text embedding vectors
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your    (x^1)
     [0.55, 0.87, 0.66],    # journey (x^2)
     [0.57, 0.85, 0.64],    # starts  (x^3)
     [0.22, 0.58, 0.33],    # with    (x^4)
     [0.77, 0.25, 0.10],    # one     (x^5)
     [0.05, 0.80, 0.55]]    # step    (x^6)
)

