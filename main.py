import pandas as pd
import os
import random
from transformers import AutoModelWithLMHead, Trainer, TrainingArguments
from processing import create_datasets, create_tokens, load_dataset


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def write_to_text(pathList):
    f = open('titles.txt', 'w')
    for path in pathList:
        df = pd.read_csv(path)
        titles = df.Title
        for i in range(len(titles)):
            print(titles[i])
            f.write('\n')
            f.write(titles[i])
    f.close()


def titles_to_list(titlePath):
    f = open('titles.txt', 'r')
    fileContent = f.readlines()
    numTitles = len(fileContent)
    flist = []
    for idx in range(numTitles):
        if fileContent[idx].endswith('\n'):
            # add to list, remove '\n'
            flist.append(fileContent[idx][:-1])

    return flist


# BEGIN MAIN SCRIPT
# check if titles already exist in a text file
# if so, create a list
if os.path.exists('titles.txt'):
    title_list = titles_to_list('titles.txt')
else:
    csvPaths = ['csv-Medicalphy-set_1974-2011.csv',
                'csv-Medicalphy-set_2013-2022.csv']
    write_to_text(csvPaths)
    title_list = titles_to_list('titles.txt')

# create a test set
num_test = 100
test_set = create_datasets(title_list, num_test)
print("break")

tokenizer = create_tokens(title_list, test_set)

model = AutoModelWithLMHead.from_pretrained('gpt2')

train_dataset, test_dataset, data_collator = load_dataset(
    "train_mod.txt", "test_mod.txt", tokenizer)

training_args = TrainingArguments(
    output_dir="gpt2_test",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=500,
    save_steps=500,
    warmup_steps=500,
    prediction_loss_only=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.save_model("gpt2_test")
tokenizer.save_pretrained("gpt2_test")
