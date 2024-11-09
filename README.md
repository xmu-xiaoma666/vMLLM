# vMLLM: Boosting Multi-modal Large Language Model with Enhanced Visual Features

This repository contains the reference code for the paper "vMLLM: Boosting Multi-modal Large Language Model with Enhanced Visual Features"

![](images/vMLLM.png)

## Experiment setup

```
# create conda envs and install corresponding package
bash running_script/install_envs.sh
```

## Data preparation
please refer to [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main), [MG-LLaVA](https://github.com/PhoenixZ810/MG-LLaVA), and [MGM](https://github.com/dvlab-research/MGM)

## Training
Training using CLIP-base vision encoder:
```
bash running_script/running_vMLLM_clip_base.sh
```

Training using CLIP-large vision encoder:
```
bash running_script/running_vMLLM_clip_large.sh
```

Training using SigLIP-base vision encoder:
```
bash running_script/running_vMLLM_siglip_base.sh
```

Training using SigLIP-SO vision encoder:
```
bash running_script/running_vMLLM_siglip_so.sh
```

## Model Zoo

| LLM | SFT  | Pretrain |
|-------|------|----------|
| Vicuna-7B  | [vMLLM_7B_sft](https://huggingface.co/xmu-xiaoma666/vMLLM_7B_sft)| [vMLLM_7B_pretrain](https://huggingface.co/xmu-xiaoma666/vMLLM_7B_pretrain)    |
| LLaMA3-8B  | [vMLLM_8B_sft](https://huggingface.co/xmu-xiaoma666/vMLLM_8B_sft) | [vMLLM_8B_pretrain](https://huggingface.co/xmu-xiaoma666/vMLLM_8B_pretrain)    |
| Vicuna-13B  | [vMLLM_13B_sft](https://huggingface.co/xmu-xiaoma666/vMLLM_13B_sft) | [vMLLM_13B_pretrain](https://huggingface.co/xmu-xiaoma666/vMLLM_13B_pretrain)     |

## Evaluation
We use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit for multi-benchmark evalution