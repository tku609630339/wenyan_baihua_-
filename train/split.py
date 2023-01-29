from datasets import load_dataset
import pandas as pd
import datasets
from datasets import Dataset
df = pd.read_excel("wenbai_73076.xlsx")
#id_list = df["id"].to_list()
#wenyan_list = df["wenyan"].to_list()
#baihua_list = df["baihua"].to_list()
#my_dict = {"id":id_list,"wenyan": wenyan_list,"baihua": baihua_list}

#dataset = Dataset.from_dict(my_dict)
dataset=Dataset.from_pandas(df)
wb_split_datasetdict = dataset.train_test_split(test_size=0.3, shuffle=True, seed=0)

train_dataset = pd.DataFrame(wb_split_datasetdict["train"])
train_length = train_dataset.shape[0]
train_dataset.to_excel(f"train_{train_length}.xlsx",index=False)
print(f"train_{train_length}.xlsx")

test_dataset = pd.DataFrame(wb_split_datasetdict["test"])
test_length = test_dataset.shape[0]
test_dataset.to_excel(f"test_{test_length}.xlsx",index=False)
print(f"test_{test_length}.xlsx")
# default loaded into 'train' fold
#data_files = {'train': 'wenbai_73076.csv'}
#wb_datasetdict = load_dataset("csv", data_files=data_files, sep=",", names=["wenyan", "baihua"])
# train fold split into train, test folds
#wb_split_datasetdict = wb_datasetdict['train'].train_test_split(test_size=0.3, shuffle=True, seed=0)
#print(type(wb_split_datasetdict))
#print(len(wb_split_datasetdict["train"]))