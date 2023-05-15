# CodeT5+

Official research release for the **CodeT5+** models (`220M`, `770M`, `2B`, `6B` `16B`) for a wide range of **Code Understanding and Generation** tasks.

*Title*: [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://github.com/salesforce/CodeT5/CodeT5+)

*Authors*: [Yue Wang](https://yuewang-cuhk.github.io/)\*, [Hung Le](https://sites.google.com/view/henryle2018/home?pli=1)\*, [Akhilesh Deepak Gotmare](https://akhileshgotmare.github.io/), [Nghi D.Q. Bui](https://bdqnghi.github.io/), [Junnan Li](https://sites.google.com/site/junnanlics), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home) (* indicates equal contribution)

# What is this about?
CodeT5+ is a new family of open code large language models with an encoder-decoder architecture that can flexibly operate in different modes (i.e. _encoder-only_, _decoder-only_, and _encoder-decoder_) to support a wide range of code understanding and generation tasks. 

To train CodeT5+, we introduce a diverse set of pretraining tasks including _span denoising_, _causal language modeling_, _contrastive learning_, and _text-code matching_ to learn rich representations from both unimodal code data and bimodal code-text data. 
Additionally, to efficiently scale up the model, we propose a simple yet effective _compute-efficient pretraining_ method to initialize our model with frozen off-the-shelf LLMs such as [CodeGen](https://github.com/salesforce/CodeGen). 
Furthermore, we explore instruction tuning to align the model with natural language instructions following [Code Alpaca](https://github.com/sahil280114/codealpaca). 

We implemented a family of CodeT5+ models, with model size ranging from 220M to 16B. 
Note that CodeT5+ 220M and 770M employ the same architecture of CodeT5-base and large respectively and are pretrained from scratch, while CodeT5+ 2B, 6B, 16B employ a "_shallow encoder and deep decoder_" architecture with the shallow encoder initialized from CodeGen-mono 350M and the deep decoder initialized from CodeGen-mono 2B, 6B, 16B, respectively.

![CodeT5+ overview](codet5p_overview.png)

# Released Models
We release the following CodeT5+ models:

* CodeT5+ `220M` and `770M` at Huggingface [here](https://huggingface.co/Salesforce/codet5p-220m) and [here](https://huggingface.co/Salesforce/codet5p-770m), respectively.
* CodeT5+ `220M` and `770M` that are further tuned on Python subset at Huggingface [here](https://huggingface.co/Salesforce/codet5p-220m-py) and [here](https://huggingface.co/Salesforce/codet5p-770m-py), respectively.
* CodeT5+ `2B`, `6B`, `16B` will be released soon.

# How to Use?
CodeT5+ `220M` and `770M` models can be easily loaded using the `T5ForConditionalGeneration` functionality. They employ the same tokenizer as the original [CodeT5](https://github.com/salesforce/CodeT5).

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "Salesforce/codet5p-770m-py"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# ==>     print('Hello World!')
```

## Citation

```bibtex
@article{wang2023codet5plus,
  title={CodeT5+: Open Code Large Language Models for Code Understanding and Generation},
  author={Wang, Yue and Le, Hung and Gotmare, Akhilesh Deepak and Bui, Nghi D.Q. and Li, Junnan and Hoi, Steven C. H.},
  journal={arXiv preprint},
  year={2023}
}
```