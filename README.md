# wenyan_baihua_-應用變形器模型於雙向文本改寫-以文言白話對照版史書為例
Using the Transformer Model for Bidirectional Text Rewriting - The Case of History Books With Aligned Classical and Modern Texts

本文使用Hugging Face平台公開的變形器模型，以網路上整理的文言及白話史書語料進行訓練，並觀察其微調前後的改寫效果，最後通過BLEU及ROUGE指標對模型生成的結果評估表現。以下列出後續將進行的4個實驗及其目的。
1.	模型微調前後比較，評估Raynardj既有模型與從Raynardj既有模型微調出最佳模型，在三冊代表性紀傳體史書，包含史記、漢書、後漢書，改寫之文言及白話生成結果。
2.	模型生成句參數修改前後比較，測試句子生成參數num_beams搜尋束個數為2、4、8與參數no_repeat_ngram_size不重複字詞數為0、2在三冊代表性紀傳體史書，包含史記、漢書、後漢書，改寫之文言及白話生成結果。
3.	不同預訓練模型比較，仿造Raynardj初始架構從頭微調出最佳模型，評估其改寫之文言及白話生成結果。
4.	將表現最佳的模型以不同資料集比較，利用本文從Raynardj微調所得最佳模型，在最佳生成參數下，評估其於另外三冊代表性紀傳體史書，包含上古後最早的晉書、中古期的新唐書、近代期的明史，改寫之文言及白話生成表現。
    為了對照雙向文本改寫實驗的表現，無論是文言文改寫白話文或是白話文改寫文言文，限定每項評測都要使用相同的參數進行。其中世代epoch表示將訓練集案例完整的餵入訓練一次，每個步驟step表示看完固定數量個批次案例的訓練量。圖11為本實驗流程圖，將從訓練集獲得最佳模型，再以訓練集與測試集預測評估生成結果。
    
    
![image](https://user-images.githubusercontent.com/73206430/215324513-f5988edc-0544-482a-ad76-4d03e3d17942.png)
