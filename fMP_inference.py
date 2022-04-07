from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
tokenizer = AutoTokenizer.from_pretrained("gpt2_test")
model = AutoModelWithLMHead.from_pretrained("gpt2_test")

gpt2_finetune = pipeline('text-generation',
                         model=model,
                         tokenizer=tokenizer)

seed = "MPnRAGE:"

test = gpt2_finetune(seed, max_length=50, num_return_sequences=5)

print("break")