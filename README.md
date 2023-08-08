# RotaryEmbedding

Side-by-side comparison of the RoPE and xPos positional embedding algorithms
used in LLMs.

This repository contains a self-contained Python implementation of each
positional embeddings.  The xPos embedding was invented as an enhancement
to RoPE, addressing problems with extrapolation and undesirable cyclical
oscillations of attention scores in the latter with increasing token distance.

The implementation of RoPE (rope.py) is borrowed from [Falcon LLM](https://huggingface.co/tiiuae/falcon-7b/blob/main/modelling_RW.py#L39)
(which in turn borrowed it from GPT-NeoX). The implementation of xPos (xpos.py) is created by changing rope.py based on the code included in [syncdoth/RetNet](https://github.com/syncdoth/RetNet)
(which in turn borrowed it from [torchscale](https://github.com/microsoft/torchscale/blob/main/torchscale/component/xpos_relative_position.py)).

Links to the original papers:
 * [RoPE](https://arxiv.org/pdf/2104.09864.pdf)
 * [xPos](https://arxiv.org/pdf/2212.10554.pdf)
