from transformers import BertModel, BertTokenizer

bert = BertModel.from_pretrained("../bert-base-chinese")

tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese")
test_sentence = "我在测试bert"
# 指定返回的数据是pytorch中的tensor数据类型
tokens = tokenizer.encode_plus(text=test_sentence, return_tensors='pt')
model_out = bert(**tokens)
print(model_out)