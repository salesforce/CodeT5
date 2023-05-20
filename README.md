# CodeT5 and CodeT5+

Official research release for  **CodeT5** and **CodeT5+** models for **Code Understanding and Generation** from Salesforce Research, which are introduced by the following papers:

*Title*: [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/pdf/2305.07922.pdf)

> *Authors*: [Yue Wang](https://yuewang-cuhk.github.io/)\*, [Hung Le](https://sites.google.com/view/henryle2018/home?pli=1)\*, [Akhilesh Deepak Gotmare](https://akhileshgotmare.github.io/), [Nghi D.Q. Bui](https://bdqnghi.github.io/), [Junnan Li](https://sites.google.com/site/junnanlics), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home) (* indicates equal contribution)

*Title*: [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf)

> *Authors*: [Yue Wang](https://yuewang-cuhk.github.io/), [Weishi Wang](https://www.linkedin.com/in/weishi-wang/)
, [Shafiq Joty](https://raihanjoty.github.io/), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home)


In practice, CodeT5 and CodeT5+ models can be deployed as an AI-powered coding assistant to boost the productivity of software developers.
At Salesforce, we build an AI coding assistant demo using CodeT5 as a VS Code plugin to provide three capabilities:

- **Text-to-code generation**: generate code based on the natural language description.
- **Code autocompletion**: complete the whole function of code given the target function name.
- **Code summarization**: generate the summary of a function in natural language description.

![CodeT5 demo](./codet5.gif)

## What's New: 🎉 

**May 2023**

**CodeT5+** paper and models are released！🔥 <br>
[paper](https://arxiv.org/pdf/2305.07922.pdf) | [code](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) | [model](https://huggingface.co/models?sort=downloads&search=codet5p) | [blog](https://blog.salesforceairesearch.com/codet5-open-code-large-language-models/)

**Sep 2022**

Our **CodeRL** paper has been accepted to NeurIPS 2022! <br>
[paper](https://arxiv.org/pdf/2207.01780.pdf) | [code](https://github.com/salesforce/CodeRL) | [blog](https://blog.salesforceairesearch.com/coderl) 


**July 2022**

We release two large-sized CodeT5 checkpoints at HuggingFace: [Salesforce/codet5-large](https://huggingface.co/Salesforce/codet5-large) and [Salesforce/codet5-large-ntp-py](https://huggingface.co/Salesforce/codet5-large-ntp-py), which are introduced by the [CodeRL paper](https://arxiv.org/pdf/2207.01780.pdf).

**Oct 2021**

We release [fine-tuned checkpoints](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/finetuned_models)
for all the downstream tasks covered in the paper.
Besides, we release a CodeT5-base fine-tuned
checkpoint ([Salesforce/codet5-base-multi-sum](https://huggingface.co/Salesforce/codet5-base-multi-sum)) for
multilingual code summarization. 


**Sep, 2021**

**CodeT5** paper accepted to EMNLP 2021 and models are released! <br>
[paper](https://arxiv.org/pdf/2109.00859.pdf) | [code](https://github.com/salesforce/CodeT5/tree/main/CodeT5) | [model](https://huggingface.co/models?sort=downloads&search=codet5) | [model card](https://github.com/salesforce/CodeT5/blob/main/CodeT5/CodeT5_model_card.pdf) | [blog](https://blog.salesforceairesearch.com/codet5/) 





## Citation

If you find this code to be useful for your research, please consider citing:

```
@inproceedings{
    wang2021codet5,
    title={CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation}, 
    author={Yue Wang, Weishi Wang, Shafiq Joty, Steven C.H. Hoi},
    booktitle={EMNLP},
    year={2021},
}

@inproceedings{
    le2022coderl,
    title={CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning},
    author={Le, Hung and Wang, Yue and Gotmare, Akhilesh Deepak and Savarese, Silvio and Hoi, Steven C. H.},
    booktitle={NeurIPS},
    year={2022}
}

@article{
    wang2023codet5plus,
    title={CodeT5+: Open Code Large Language Models for Code Understanding and Generation},
    author={Wang, Yue and Le, Hung and Gotmare, Akhilesh Deepak and Bui, Nghi D.Q. and Li, Junnan and Hoi, Steven C. H.},
    journal={arXiv preprint},
    year={2023}
}
```

## License

The code is released under the BSD-3 License (see `LICENSE.txt` for details), but we also ask that users respect the
following:

This software should not be used to promote or profit from:

violence, hate, and division,

environmental destruction,

abuse of human rights, or

the destruction of people's physical and mental health.

We encourage users of this software to tell us about the applications in which they are putting it to use by emailing
codeT5@salesforce.com, and to
use [appropriate](https://arxiv.org/abs/1810.03993) [documentation](https://www.partnershiponai.org/about-ml/) when
developing high-stakes applications of this model.


## Get Involved

Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. We welcome PRs!

