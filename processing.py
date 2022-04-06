import random
from transformers import AutoTokenizer, AutoModelWithLMHead, TextDataset, DataCollatorForLanguageModeling


def create_datasets(title_list, num_test):
    # create a test set
    test_set = []
    for idx in range(num_test):
        test_idx = random.choice(range(len(title_list)))
        test_set.append(title_list[test_idx])
        title_list.pop(test_idx)

    return test_set


def create_tokens(title_list, test_set):

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # add tokens to training dataset
    train_m = ""
    for title in title_list:
        train_m += (tokenizer.special_tokens_map['bos_token'] + title.rstrip() + tokenizer.special_tokens_map[
            'eos_token'])
    with open("train_mod.txt", "w", encoding='utf-8') as f:
        f.write(train_m)
    # add tokens to testing dataset
    test_m = ""
    for title in test_set:
        test_m += (tokenizer.special_tokens_map['bos_token'] + title.rstrip() + tokenizer.special_tokens_map[
            'eos_token'])
    with open("test_mod.txt", "w", encoding='utf-8') as f:
        f.write(test_m)

    return tokenizer

def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(tokenizer=tokenizer,
                                file_path=train_path,
                                block_size=128)
    test_dataset = TextDataset(tokenizer=tokenizer,
                               file_path=test_path,
                               block_size=128)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
    return train_dataset, test_dataset, data_collator


