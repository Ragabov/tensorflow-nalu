# Tensorflow NALU

A tensorflow implementation of Neural Arithmetic Logic Unit, Trask et al.
Paper: https://arxiv.org/abs/1808.00508

## Getting Started

```python 
from nalu.layers.nalu_layer import NaluLayer

nalu_layer = NaluLayer(FEATURES_NUM, OUTPUT_DIM, HIDDEN_DIM, N_CELLS, CORE_CELL_TYPE)
outputs = nalu_layer(input)
``` 
