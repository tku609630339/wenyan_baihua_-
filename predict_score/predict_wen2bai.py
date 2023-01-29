
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

from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.cuda.set_device(1)

def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]

def predict_wen2bai(dataset, model, encoder_tokenizer, decoder_tokenizer,
                               batch_size=16,  device=device,
                               column_source="wenyan", 
                               column_target="baihua"):
    source_batches = list(chunks(dataset[column_source], batch_size))
    target_batches = list(chunks(dataset[column_target], batch_size))
    target_list=[]
    for source_batch, target_batch in tqdm(
        zip(source_batches, target_batches), total=len(source_batches)):
        
        inputs = encoder_tokenizer(source_batch, max_length=512,  truncation=True, 
                        padding="max_length", return_tensors="pt")
        
        targets = model.generate(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device), 
                         length_penalty=0.8, num_beams=8, no_repeat_ngram_size=1,
                                 max_length=256)
        
        decoded_targets = [decoder_tokenizer.decode(s, skip_special_tokens=True, 
                                clean_up_tokenization_spaces=True) 
               for s in targets]
        decoded_targets = [d.replace(" ", "") for d in decoded_targets]
        target_list.extend(decoded_targets)
    return target_list



PRETRAINED = "best_w2b_min"


GPT2_PRETRAINED = 'uer/gpt2-chinese-poem'
tokenizer_decoder = BertTokenizerFast.from_pretrained(GPT2_PRETRAINED)
#vocab_decoder = tokenizer_decoder.get_vocab()

tokenizer_encoder = BertTokenizer.from_pretrained(PRETRAINED)
#vocab_encoder = tokenizer_encoder.get_vocab()
model = EncoderDecoderModel.from_pretrained(PRETRAINED).to(device)
#seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer_encoder, model=model)

model.config.decoder_start_token_id = tokenizer_decoder.cls_token_id
model.config.pad_token_id = tokenizer_decoder.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

train_df = pd.read_excel("train_51153.xlsx")
#test_df = pd.read_excel("test_21923.xlsx")
train_dataset = Dataset.from_pandas(train_df)
#test_dataset = Dataset.from_pandas(test_df)
#wb_split_datasetdict = datasets.DatasetDict({"train":train_dataset})

#dataset=wb_split_datasetdict['train']

from datasets import Dataset
import pandas as pd
id1=train_dataset["id"]
wenyan=train_dataset["wenyan"]
baihua=train_dataset["baihua"]
df=pd.DataFrame({"id":id1,"baihua":baihua,"wenyan":wenyan})
df.sort_values(by="wenyan",key=lambda x: x.str.len(),ascending=True,inplace=True)
dataset=Dataset.from_pandas(df)

target_list=predict_wen2bai(dataset, model, tokenizer_encoder, tokenizer_decoder,
                               batch_size=4,  device=device,
                               column_source="wenyan", 
                               column_target="baihua")

length=len(dataset)
f_name=f"raynardj_w2b_nb8_min_{length}.csv"
df=pd.DataFrame({"id":dataset["id"],  "wenyan":dataset["wenyan"], "baihua":dataset["baihua"], "predict_baihua":target_list})
df.to_excel(f_name,index=False)
print(f'{f_name}: shape={df.shape}\n')