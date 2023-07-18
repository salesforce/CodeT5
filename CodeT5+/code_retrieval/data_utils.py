import json
import torch
from torch.utils.data import DataLoader, Dataset


def create_dataset(data_dir, task):
    if task == 'AdvTest':
        train_dataset = csn_search_train(data_dir, task, 'train')
        val_dataset = advtest_search_eval_text(data_dir, task, 'valid')
        test_dataset = advtest_search_eval_text(data_dir, task, 'test')
        codebase_dataset = csn_search_eval_code(data_dir, task, 'test.jsonl')
        return train_dataset, val_dataset, test_dataset, codebase_dataset
    elif task == 'cosqa':
        train_dataset = cosqa_search_train(data_dir, task, 'cosqa-retrieval-train-19604.json')
        val_dataset = cosqa_search_eval_text(data_dir, task, 'cosqa-retrieval-dev-500.json')
        test_dataset = cosqa_search_eval_text(data_dir, task, 'cosqa-retrieval-test-500.json')
        codebase_dataset = cosqa_search_eval_code(data_dir, task)
        return train_dataset, val_dataset, test_dataset, codebase_dataset
    else:
        train_dataset = csn_search_train(data_dir, task, 'train')
        val_dataset = csn_search_eval_text(data_dir, task, 'valid')
        test_dataset = csn_search_eval_text(data_dir, task, 'test')
        codebase_dataset = csn_search_eval_code(data_dir, task, 'codebase.jsonl')
        return train_dataset, val_dataset, test_dataset, codebase_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 text,
                 code,
                 url=None
                 ):
        self.idx = idx
        self.text = text
        self.code = code
        self.url = url


# for notice, in case this will cause errors
def replace_special_tokens(line):
    return line.replace('<pad>', '</pad>').replace('<s>', '<ss>').replace('</s>', '</ss>')


def read_search_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            if 'function_tokens' in js:
                js['code_tokens'] = js['function_tokens']
            code = replace_special_tokens(' '.join(js['code_tokens']))
            nl = replace_special_tokens(' '.join(js['docstring_tokens']))
            examples.append(
                Example(
                    idx=idx,
                    text=nl,
                    code=code,
                    url=js['url']
                )
            )

    print(f'Read {len(examples)} data from {filename}')
    return examples


def read_cosqa_search_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        if "code_idx_map" in filename:
            js = json.load(f)
            for key in js:
                examples.append(
                    Example(
                        idx=js[key],
                        text="",
                        code=key,
                        url=js[key]
                    )
                )
        else:
            data = json.load(f)
            for idx, js in enumerate(data):
                code = replace_special_tokens(' '.join(js['code_tokens'].split()))
                nl = replace_special_tokens(' '.join(js['doc'].split()))
                examples.append(
                    Example(
                        idx=idx,
                        text=nl,
                        code=code,
                        url=js['retrieval_idx']
                    )
                )

    print(f'Read {len(examples)} data from {filename}')
    return examples


class csn_search_train(Dataset):
    def __init__(self, data_dir, lang, split='train'):
        self.examples = read_search_examples(f'{data_dir}/{lang}/{split}.jsonl')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ex = self.examples[index]
        return ex.text, ex.code, ex.idx


class csn_search_eval_text(Dataset):
    def __init__(self, data_dir, lang, split='valid'):
        self.examples = read_search_examples(f'{data_dir}/{lang}/{split}.jsonl')
        self.codebase = read_search_examples(f'{data_dir}/{lang}/codebase.jsonl')

        self.text = []
        self.code = []

        text2url = {}
        url2code = {}

        for idx, ex in enumerate(self.examples):
            self.text.append(ex.text)
            text2url[idx] = ex.url

        for idx, ex in enumerate(self.codebase):
            self.code.append(ex.code)
            url2code[ex.url] = idx

        self.text2code = {}

        for text_id, text in enumerate(self.text):
            self.text2code[text_id] = url2code[text2url[text_id]]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index]


class advtest_search_eval_text(Dataset):
    def __init__(self, data_dir, lang, split='valid'):
        self.examples = read_search_examples(f'{data_dir}/{lang}/{split}.jsonl')

        # below is for advtest
        self.text2code = {}
        for ex in self.examples:
            self.text2code[ex.idx] = ex.idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].text


class csn_search_eval_code(Dataset):
    def __init__(self, data_dir, lang, codebase_fn='codebase.jsonl'):
        self.code = [ex.code for ex in read_search_examples(f'{data_dir}/{lang}/{codebase_fn}')]

    def __len__(self):
        return len(self.code)

    def __getitem__(self, index):
        return self.code[index]


class cosqa_search_train(Dataset):
    def __init__(self, data_dir, lang, split='train'):
        self.examples = read_cosqa_search_examples(f'{data_dir}/{lang}/{split}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ex = self.examples[index]
        return ex.text, ex.code, ex.idx


class cosqa_search_eval_text(Dataset):
    def __init__(self, data_dir, lang, split='valid'):
        self.examples = read_cosqa_search_examples(f'{data_dir}/{lang}/{split}')
        self.codebase = read_cosqa_search_examples(f'{data_dir}/{lang}/code_idx_map.txt')

        self.text = []
        self.code = []

        text2url = {}
        url2code = {}

        for idx, ex in enumerate(self.examples):
            self.text.append(ex.text)
            text2url[idx] = ex.url

        for idx, ex in enumerate(self.codebase):
            self.code.append(ex.code)
            url2code[ex.url] = idx

        self.text2code = {}

        for text_id, text in enumerate(self.text):
            self.text2code[text_id] = url2code[text2url[text_id]]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index]


class cosqa_search_eval_code(Dataset):
    def __init__(self, data_dir, lang):
        self.code = [ex.code for ex in read_cosqa_search_examples(f'{data_dir}/{lang}/code_idx_map.txt')]

    def __len__(self):
        return len(self.code)

    def __getitem__(self, index):
        return self.code[index]
