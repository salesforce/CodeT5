from tokenizers import ByteLevelBPETokenizer

paths = ['train_code.txt', 'train_doc.txt']

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=32000, min_frequency=3, special_tokens=[
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<mask>"
])

# Save files to disk
tokenizer.save_model("./salesforce", "codet5")

print(
    tokenizer.encode("<s> hello <unk> Don't you love 🤗 Transformers <mask> yes . </s>").tokens
)
