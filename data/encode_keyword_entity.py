import os
import pickle
from sentence_transformers import SentenceTransformer, models

# 设置代理（如有需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

# 加载原始 HuggingFace 格式的本地模型路径
hf_model_path = r"D:\科研\mindmap\MindMap-main - 副本 - 副本\MindMap-main\embedding_model\pubmedbert"

# 封装成 sentence-transformers 模型（只需执行一次）
word_embedding_model = models.Transformer(hf_model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 读取实体数据
with open("entity2id.txt", "r", encoding="utf-8") as f:
    entities = f.readlines()
    entities = [entity.strip().split()[0].replace("_", " ") for entity in entities]

# keywords 可填关键词集合，如果没有就留空
keywords = set([])

# 编码实体
embeddings = model.encode(entities, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
entity_emb_dict = {
    "entities": entities,
    "embeddings": embeddings,
}

# 保存实体向量
with open("entity_embeddings_pubmedbert.pkl", "wb") as f:
    print(entity_emb_dict.keys())  # 输出检查
    pickle.dump(entity_emb_dict, f)

# 编码关键词（如果为空也不会报错）
embeddings = model.encode(list(keywords), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
keyword_emb_dict = {
    "keywords": list(keywords),
    "embeddings": embeddings,
}

# 保存关键词向量
with open("keyword_embeddings_pubmedbert.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)

print("✅ 所有嵌入完成，保存成功！")
