from datasets import load_dataset

from transformers import (
    EncoderDecoderModel,
    AutoTokenizer,
    pipeline
)
import pandas as pd
import datasets
from datasets import Dataset
from transformers import BertTokenizer, BertTokenizerFast
from transformers import Trainer, TrainingArguments,Seq2SeqTrainer,Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
#torch.cuda.set_device(2)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# count = 0

def convert_examples_to_features(example_batch):
    input_encodings = tokenizer_encoder(example_batch["baihua"], max_length=1024,
                                        truncation=True)

    with tokenizer_decoder.as_target_tokenizer():
        target_encodings = tokenizer_decoder(example_batch["wenyan"], max_length=1024,
                                             truncation=True)

    print(f'{example_batch["baihua"][0]}\n')
    # count = count + 1

    return {"input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]}
            #"decoder_input_ids": target_encodings["input_ids"]}



def inference(text):
    tk_kwargs = dict(
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors='pt')

    inputs = tokenizer_encoder([text, ], **tk_kwargs)
    with torch.no_grad():
        return tokenizer_decoder.batch_decode(
            model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=4,
                no_repeat_ngram_size=2,
                max_length=256,
                bos_token_id=101,
                eos_token_id=tokenizer_decoder.sep_token_id,
                pad_token_id=tokenizer_decoder.pad_token_id,
            ), skip_special_tokens=True)


# PRETRAINED = "Helsinki-NLP/opus-mt-fr-en"
# MarianMTModel = MarianEncoder + MarianDecoder
# translator = pipeline("translation", model=PRETRAINED)
# translator("Ce cours est produit par Hugging Face.")

# https://huggingface.co/raynardj/wenyanwen-ancient-translate-to-modern
# EncoderDecoderModel = BertForMaskedLM (bert-base-chinese:21128) + GPT2LMHeadModel (uer/gpt2-chinese-poem:22557)
GPT2_PRETRAINED = 'uer/gpt2-chinese-poem'
# pipe = pipeline('text-generation', GPT2_PRETRAINED)
# pipe_out = pipe('??????????????????????????????')
tokenizer_decoder = BertTokenizerFast.from_pretrained(GPT2_PRETRAINED)
vocab_decoder = tokenizer_decoder.get_vocab()
print(f'vocab_decoder: {len(vocab_decoder)}')

PRETRAINED = "raynardj/wenyanwen-chinese-translate-to-ancient"#????????????????????????????????????
# AttributeError: 'GPT2Config' object has no attribute '_AutoTokenizer__class'
# tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
tokenizer_encoder = BertTokenizer.from_pretrained(PRETRAINED)
vocab_encoder = tokenizer_encoder.get_vocab()
print(f'vocab_encoder: {len(vocab_encoder)}')
# tokenizer_decoder =
model = EncoderDecoderModel.from_pretrained(PRETRAINED)#.to(device)
#seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer_encoder, model=model)

model.config.decoder_start_token_id = tokenizer_decoder.cls_token_id
model.config.pad_token_id = tokenizer_decoder.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model_name = 'raynardj/'

baihua = '?????????????????????????????????'
# input_ids = tokenizer_encoder("???????????????????????????", return_tensors="pt").input_ids
# labels = tokenizer_decoder("?????????????????????????????????", return_tensors="pt").input_ids
# outputs = model(input_ids=input_ids, labels=labels)
# loss, logits = outputs.loss, outputs.logits
wenyan = inference(baihua)
print(wenyan)

train_df = pd.read_excel("train_51153.xlsx")
test_df = pd.read_excel("test_21923.xlsx")
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
wb_split_datasetdict = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})


# hide_output
wb_encoded_split_datasetdict = wb_split_datasetdict.map(convert_examples_to_features, batched=True,
                                                        batch_size=1000)  # default 1000
columns = ["input_ids", "attention_mask", "labels"]
wb_encoded_split_datasetdict.set_format(type="torch", columns=columns)

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer_encoder, model=model)

#torch.cuda.set_device(3)
# emotions_encoded = emotions_local.map(tokenize, batched=True, batch_size=None)
batch_size = 2
logging_steps = 10
training_args = Seq2SeqTrainingArguments(
    output_dir='bai2wen_min', num_train_epochs=3, warmup_steps=500,
    per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
    weight_decay=0.01, logging_steps=logging_steps, push_to_hub=False,
    evaluation_strategy='epoch',save_strategy='epoch',# eval_steps=1000, save_steps=1000,
    save_total_limit = 2,load_best_model_at_end=True,
    gradient_accumulation_steps=16)


trainer = Seq2SeqTrainer(model=model, args=training_args,
                  tokenizer=tokenizer_encoder, 
                  data_collator=seq2seq_data_collator,
                  train_dataset=wb_encoded_split_datasetdict["train"],
                  eval_dataset=wb_encoded_split_datasetdict["test"])

trainer.train()#resume_from_checkpoint=True)
# rest of the training args
# ...
#training_args.logging_dir = 'logs' # or any dir you want to save logs
best_ckpt_path = trainer.state.best_model_checkpoint
print("??????????????????:",best_ckpt_path)
f=open('b2w_out.txt','w')
print("??????????????????:",best_ckpt_path,file=f)
f.close()
trainer.save_model('best_b2w_min')
trainer.save_state()#trainer_state.json