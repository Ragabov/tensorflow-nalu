# Tensorflow NALU

A tensorflow implementation of Neural Arithmetic Logic Unit, Trask et al.
Paper: https://arxiv.org/abs/1808.00508

## Modifications to the original model 

The NALU cell proposed in the paper had a major drawback that it can't model multiplication of negative inputs since 
multiplication is implemented as addition in log-space. 

A solution to this was to avoid working in the log-space of the inputs and instead use the asinh function as porposed here:
https://www.reddit.com/r/MachineLearning/comments/94833t/neural_arithmetic_logic_units/e3u974x/

## Getting Started

```python 
from nalu.layers.nalu_layer import NaluLayer

nalu_layer = NaluLayer(FEATURES_NUM, OUTPUT_DIM, HIDDEN_DIM, N_CELLS, CORE_CELL_TYPE)
outputs = nalu_layer(input)
``` 
