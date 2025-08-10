from transformers import AutoTokenizer, AutoModel
import pickle
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
# 设置HTTPS代理，同上
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

entity_embeddings = {}
with open('entity2id.txt', 'r') as file:
    for line in file:
        entity, _ = line.strip().split('\t')
        inputs = tokenizer(entity, return_tensors="pt")
        outputs = model(**inputs)
        # 取最后一层的输出并计算平均值作为嵌入向量
        embeddings = outputs.last_hidden_state.mean(dim=1)
        entity_embeddings[entity] = embeddings.detach().numpy()


# 将嵌入向量保存到 pickle 文件中
with open('1.pkl', 'wb') as pkl_file:
    pickle.dump(entity_embeddings, pkl_file)