# Self attention is a mechanism to parse the input data 
# Will proceed with simplified self attention -> self-attention -> causal self-attention -> Multi-head attention (adv causal attention)

# Why care about self-attention? RNNs were standard before, which stored the input in latent memory while generating the output. Called memory cell/hidden state
# The memory cell approach lost some information which impacted performance

# Attention allows selective access of the actual input tokens, not just a representation of them, which improves performance by not loosing info

# There are 'importance/attention scores/attention weights' which assign a relative importance to each input token

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

input_query = inputs[1]     # using the second 'token' journey as an example

# dot is element wise multiplication then addition

# python for loop
# result = 0
# i = 1

# for idx, element in enumerate(inputs[i]):
#     result += inputs[i][idx] * input_query[idx]
# print(result)

#pytorch dot product for loop
query = inputs[1]               # all this is in respect to token 2

attn_scores = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i, input_query) # dot product (transpose not necessary here since they are 1-dim vectors)

# Now have to normalise the attention scores so they sum to 1 - this gives us the attention weights
# could use:
# attn_scores_tmp = attn_scores / attn_scores.sum()

# better to use softmax (just a normalisation function)
# def softmax_naive(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)

# print(softmax_naive(attn_scores))

# pytorch version of softmax: this gives the attention weights alpha
attn_weights = torch.softmax(attn_scores, dim=0)

# ----------------- 
# Now compute the context vector by applying the attention weights from the softmax to the input embedding vectors
# context vector summarises all the input from the other tokens in the input - higher weights = more significance to the output
# ----------------- 

context_vec = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec += attn_weights[i] * x_i

# print(context_vec)

# ----------------- 
# Computing Attention weights for all input tokens
# ----------------- 

