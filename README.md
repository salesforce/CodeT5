# CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation
This is the official PyTorch implementation for the following EMNLP 2021 paper from Salesforce Research: 

**Title**: [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf) 

**Authors**: [Yue Wang](https://yuewang-cuhk.github.io/), [Weishi Wang](https://www.linkedin.com/in/weishi-wang/), [Shafiq Joty](https://raihanjoty.github.io/), and [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home) 

![CodeT5 demo](codet5.gif)

## Updates
**Sep 24, 2021**

CodeT5 is now in [hugginface](https://huggingface.co/)!

You can simply load the model ([CodeT5-small](https://huggingface.co/Salesforce/codet5-small) and [CodeT5-base](https://huggingface.co/Salesforce/codet5-base)) and do the inference:

```python
from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

text = "def greet(user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate one code span
generated_ids = model.generate(input_ids, max_length=8)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "{user.username}"
```

## Introduction
This repo provides the code for reproducing the experiments in [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf). 
CodeT5 is a new pre-trained encoder-decoder model for programming languages, which is pre-trained on 8.35M functions in 8 programming languages (Python, Java, JavaScript, PHP, Ruby, Go, C, and C#). 
In total, it achieves state-of-the-art results on 14 sub-tasks in a code intelligence benchmark - [CodeXGLUE](https://github.com/microsoft/CodeXGLUE). 

Paper link: https://arxiv.org/abs/2109.00859

Blog link: https://blog.einstein.ai/codet5/

The code currently include two pre-trained checkpoints ([CodeT5-small](https://huggingface.co/Salesforce/codet5-small) and [CodeT5-base](https://huggingface.co/Salesforce/codet5-base)) and scripts to fine-tine them on 4 generation tasks (code summarization, code generation, translation, and refinement) plus 2 understanding tasks (code defect detection and clone detection) in CodeXGLUE.

In practice, CodeT5 can be deployed as an AI-powered coding assistant to boost the productivity of software developers. 
At Salesforce, we build an [AI coding assistant demo](https://github.com/salesforce/CodeT5/raw/main/codet5.gif) using CodeT5 to provide three capabilities for Apex developers as a VS Code plugin:

- **Text-to-code generation**: generate code based on the natural language description.
- **Code autocompletion**: complete the whole function of code given the target function name.
- **Code summarization**: generate the summary of a function in natural language description.  

## Table of Contents

1. [Citation](#citation)
2. [License](#license)
3. [Dependency](#dependency)
4. [Download](#download)
5. [Fine-tuning](#fine-tuning)
6. [Get Involved](#get-involved)

## Citation
If you find this code to be useful for your research, please consider citing.
```
@article{CodeT5,
      title={CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation}, 
      author={Yue Wang, Weishi Wang, Shafiq Joty, Steven C.H. Hoi},
      year={2021},
      journal={arXiv preprint arXiv:2109.00859},
}
```

## License
The code is released under the BSD-3 License (see `LICENSE.txt` for details), but we also ask that users respect the following:

This software should not be used to promote or profit from:

violence, hate, and division,

environmental destruction,

abuse of human rights, or 

the destruction of people's physical and mental health.

We encourage users of this software to tell us about the applications in which they are putting it to use by emailing codeT5@salesforce.com, and to use [appropriate](https://arxiv.org/abs/1810.03993) [documentation](https://www.partnershiponai.org/about-ml/) when developing high-stakes applications of this model.

## Dependency
- Pytorch 1.7.1
- tensorboard 2.4.1
- transformers 4.6.1
- tree-sitter 0.2.2 
 
## Download 
* [Pre-trained checkpoints & Fine-tuning data](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research)

Instructions for download:
```
pip install gsutil

gsutil -m cp -r \
  "gs://sfr-codet5-data-research/data/" \
  "gs://sfr-codet5-data-research/pretrained_models/" \
  .
```

The repository structure is shown in the following after download:
```
├── CODE_OF_CONDUCT.md
├── README.md
├── SECURITY.md
├── codet5.gif
├── configs.py
├── models.py
├── run_clone.py
├── run_gen.py
├── utils.py
├── _utils.py
├── LICENSE.txt
├── data
│   ├── clone
│   ├── concode
│   ├── defect
│   ├── refine
│   │   ├── medium
│   │   └── small
│   ├── summarize
│   │   ├── go
│   │   ├── java
│   │   ├── javascript
│   │   ├── php
│   │   ├── python
│   │   └── ruby
│   └── translate
├── evaluator
│   ├── bleu.py
│   ├── smooth_bleu.py
│   └── CodeBLEU
├── pretrained_models
│   ├── codet5_base
│   └── codet5_small
├── sh
│   ├── exp_with_args.sh
│   ├── run_exp.py
│   ├── results
│   ├── saved_models
│   └── tensorboard
└── tokenizer
    └── salesforce
        ├── codet5-merges.txt
        └── codet5-vocab.json    
```

## Fine-tuning
Go to `sh` folder, set the `WORKDIR` in `exp_with_args.sh` to be your downloaded CodeT5 repository path.
 
You can use `run_exp.py` to run a broad set of experiments by simply passing the `model_tag`, `task`, and `sub_task` arguments. 
In total, we support four models (i.e., ['roberta', 'codebert', 'codet5_small', 'codet5_base']) and six tasks (i.e., ['summarize', 'concode', 'translate', 'refine', 'defect', 'clone']). 
For each task, we use the `sub_task` to specify which specific datasets to fine-tine on.
 
For example, if you want to run CodeT5-base model on the code summarization task for Ruby, you can simply run:
```
python run_exp.py --model_tag codet5_base --task summarize --sub_task ruby
```

Besides, you can specify:
```
model_dir: where to save fine-tuning checkpoints
res_dir: where to save the performance results 
summary_dir: where to save the training curves
data_num: how many data instances to use, the default -1 is for using the full data
gpu: the index of the GPU to use in the cluster
``` 
You can also directly revise the suggested arguments in the [get_args_by_task_model](https://github.com/salesforce/CodeT5/blob/4f8818aea1bf170f019381671087e4c4f9608005/sh/run_exp.py#L14) function. 
Please refer to the argument flags in `configs.py` for the full available options.
The saved training curves in `summary_dir` can be visualized using [tensorboard](https://pypi.org/project/tensorboard/).

## Get Involved

Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. 
We welcome PRs!

