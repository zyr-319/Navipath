from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import heapq
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import itertools
from typing import Dict, List
import pickle
import json
import openai
import os
from nltk.tokenize import word_tokenize
import csv
from sentence_transformers import SentenceTransformer,models

os.environ['HTTP_PROXY'] = 'http://172.29.13.107:5782'
# 设置HTTPS代理 ，同上
os.environ['HTTPS_PROXY'] = 'http://172.29.13.107:5782'

# 加载模型
model_path = "embedding_model/MedBERT"
word_embedding_model = models.Transformer(model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def align_entities_to_kg_entities(question_kg, similarity_f):
    """
    使用中文嵌入模型将输入实体与知识图谱中的实体通过语义相似度进行匹配。
    如果匹配实体与原实体重合字符少于2个，则重新找重合字符最多且至少为2的实体。
    """
    match_kg = []

    all_kg_entities = entity_embeddings["entities"]
    all_kg_embeddings = np.array(entity_embeddings["embeddings"])

    for kg_entity in question_kg:
        question_emb = embedding_model.encode(kg_entity)
        similarities = cosine_similarity_manual(all_kg_embeddings, question_emb.reshape(1, -1)).flatten()

        max_index = np.argmax(similarities)
        max_similarity = similarities[max_index]
        matched_entity = all_kg_entities[max_index]

        if max_similarity >= similarity_f :
            matched_entity = matched_entity.replace(" ", "_")
            if matched_entity not in match_kg:
                match_kg.append(matched_entity)

    return match_kg


def get_shortest_path_length(entity1, entity2):
    query = """
    MATCH (start:Entity {name: $entity1}), (end:Entity {name: $entity2}),
    p = shortestPath((start)-[*..5]-(end))
    RETURN length(p) AS path_length
    """
    try:
        result = session.run(query, entity1=entity1.replace("_", " "), entity2=entity2.replace("_", " "))
        record = result.single()
        return record["path_length"] if record else None
    except:
        return None


def cot_expand_entities_batch(question_text):
    """
    单条 Prompt 实现，保留原始指令并加入一个实体列表示例，提升格式稳定性。
    """

    prompt = f"""
You are a professional and knowledgeable medical assistant. You will be given a medical question. 
Please think step by step, identify key medical clues in the question, and infer related medical concepts, including diseases, symptoms, diagnostic tests, and treatments. Avoid duplication.

In the last paragraph, output 15 to 20 related medical entities that could help solve the question. Separate entities with commas. Use underscores (_) to join words in multi-word terms. 
Only the final paragraph should contain the entity list.

Question: {question_text}

""".strip()

    response = chat([HumanMessage(content=prompt)], model="gpt-3.5-turbo")
    raw_output = response.content.strip()

    lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
    raw_entities = lines[-1]
    expanded_entities = [e.strip().replace("_", " ") for e in raw_entities.split(",") if e.strip()]
    return expanded_entities




def cluster_candidates(candidate_paths, scores, num_clusters):
    """
    对 candidate_paths 进行结构去重 + 聚类，并从每个簇中选取得分最高路径。

    参数：
        candidate_paths: list[str]，候选路径的文本列表
        scores: list[float]，对应每条路径的得分
        num_clusters: int，聚类数

    返回：
        representative_candidates: list[str]，每个聚类中选取的代表路径
    """

    if not candidate_paths:
        return []

    # Step 1: 基于实体序列结构去重
    seen_structures = set()
    unique_paths = []
    unique_scores = []

    for path, score in zip(candidate_paths, scores):
        entities = tuple([x.strip() for i, x in enumerate(path.split("->")) if i % 2 == 0])
        if entities not in seen_structures:
            seen_structures.add(entities)
            unique_paths.append(path)
            unique_scores.append(score)

    # Step 2: 如果数量比聚类数还少，直接返回
    if len(unique_paths) <= num_clusters:
        return unique_paths

    # Step 3: 文本向量化
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 或其他 SBERT 模型
    X = model.encode(unique_paths)

    # Step 4: 聚类
    clustering = AgglomerativeClustering(n_clusters=num_clusters)
    labels = clustering.fit_predict(X)

    # Step 5: 每个簇中选分数最高的路径
    cluster_dict = {}
    for label, path, score in zip(labels, unique_paths, unique_scores):
        cluster_dict.setdefault(label, []).append((path, score))

    representative_candidates = []
    for cluster_paths in cluster_dict.values():
        best_path = max(cluster_paths, key=lambda x: x[1])
        representative_candidates.append(best_path[0])

    return representative_candidates


def get_shortest_path_length(origin, candidate, session, max_hops=3):
    """
    查询 Neo4j 知识图谱中从 origin 到 candidate 的最短路径跳数，
    如果不存在路径则返回 None。
    """
    query = (
        "MATCH (o:Entity {name:$origin}), (c:Entity {name:$candidate}) "
        f"MATCH p = shortestPath((o)-[*..{max_hops}]-(c)) "
        "RETURN length(p) AS hops"
    )
    result = session.run(query, origin=origin, candidate=candidate)
    record = result.single()
    if record is None:
        return None
    return record["hops"]


def score_path_with_lookahead(current_entity, path, depth, question, max_lookahead, visited_entities=None):
    """
    计算路径的综合得分，包括未来 max_lookahead 步内的潜在得分，并返回路径和得分。
    如果达到最大深度则停止递归。
    - visited_entities 用于避免路径循环。
    """
    if visited_entities is None:
        visited_entities = set()  # 如果没有传入 visited_entities，则初始化为空集合

    # 检查当前实体是否已经出现在路径中，避免循环
    if current_entity in visited_entities:
        return path, -1  # 如果当前实体已经在路径中，返回负得分（避免这条路径）

    # 将当前实体添加到已访问实体集合中
    visited_entities.add(current_entity)

    # 将路径元素转为字符串格式
    path_str_elements = [f"{rel} -> {ent}" for rel, ent in path]
    current_path_str = " -> ".join(path_str_elements)  # 使用完整路径创建路径字符串

    # Tokenize 查询和路径
    tokenized_query = " ".join(word_tokenize(question))
    tokenized_path = " ".join(word_tokenize(current_path_str))

    # 提取特征并计算当前路径得分
    features_df = extract_features(tokenized_query, [tokenized_path])
    current_score = score_path(features_df.iloc[0])

    # 如果达到最大 lookahead 深度，返回当前路径和当前得分
    if depth >= max_lookahead:
        # print(f"Max depth reached: {current_path_str}, Score: {current_score}")
        return path, current_score

    # 获取当前实体的邻居
    neighbors = lookahead_get_entity_neighbors(current_entity)
    if not neighbors:
        return path, current_score

    future_scores = []
    for relation, neighbor in neighbors:
        new_path = path + [(relation, neighbor)]
        # 复制当前的 visited_entities 集合，以避免不同路径之间的干扰
        new_visited_entities = visited_entities.copy()
        _, future_score = score_path_with_lookahead(neighbor, new_path, depth + 1, question, max_lookahead, new_visited_entities)
        if future_score >= 0:  # 确保路径没有被剪枝
            future_scores.append(future_score)

    # 计算未来潜在得分，并计算路径的总得分
    average_future_score = sum(future_scores) / len(future_scores) if future_scores else 0
    total_score = current_score + average_future_score

    # print(f"Path: {current_path_str}, Current Score: {current_score}, Future Potential Score: {average_future_score}, Total Score: {total_score}")

    return path, total_score


def top_k_paths_with_lookahead(initial_entity, question, max_depth, max_lookahead, max_paths):
    queue = []
    # 初始路径包含初始实体，关系设置为 None
    initial_path = [(None, initial_entity)]

    # 计算初始路径的得分
    _, initial_score = score_path_with_lookahead(initial_entity, initial_path, 0, question, max_lookahead)
    heapq.heappush(queue, (-initial_score, initial_path))  # 使用负分数保证最大堆行为

    # 存储最终结果
    top_k_results = []

    # 跟踪已访问的实体，避免重复路径
    visited_entities_set = set([initial_entity])

    # 使用一个缓存字典来存储实体的邻居，避免重复查询
    neighbor_cache = {}

    while queue and len(top_k_results) < max_paths:
        # 从队列中弹出当前分数最高的路径
        current_score, current_path = heapq.heappop(queue)
        current_entity = current_path[-1][1]  # 当前路径的最后一个实体
        current_depth = len(current_path) - 1  # 当前深度

        # 检查是否达到了最大深度限制
        if current_depth >= max_depth:
            # print(f"Reached max depth with path: {current_path}, Score: {current_score}")
            top_k_results.append((current_path, -current_score))
            continue

        # 获取当前实体的直接邻居（使用缓存避免重复查询）
        if current_entity not in neighbor_cache:
            neighbors = lookahead_get_entity_neighbors(current_entity)
            neighbor_cache[current_entity] = neighbors
        else:
            neighbors = neighbor_cache[current_entity]

        # 如果没有邻居，跳过这条路径
        if not neighbors:
            # print(f"No neighbors found for entity: {current_entity}")
            continue

        # 跟踪访问的实体，避免循环路径
        visited_entities = set(ent for _, ent in current_path)

        for relation, neighbor in neighbors:
            # 如果邻居已经访问过，跳过
            if neighbor in visited_entities:
                # print(f"Skipping neighbor {neighbor} to avoid loop.")
                continue

            # 如果邻居是路径起点，跳过，避免循环路径
            if neighbor == current_path[0][1]:
                # print(f"Skipping path ending at {neighbor} to avoid circular path.")
                continue

            # 创建新的扩展路径，包含此前路径的完整信息
            new_path = current_path + [(relation, neighbor)]

            # 计算新路径的得分
            _, lookahead_score = score_path_with_lookahead(neighbor, new_path, current_depth + 1, question, max_lookahead)
            total_score = -current_score + lookahead_score
            # print(f"Extended Path: {new_path}, Total Score: {current_score} + {lookahead_score} = {total_score}")

            # 将新路径加入优先级队列
            heapq.heappush(queue, (-total_score, new_path))

    # 如果 top_5_results 不满 5 条路径，从队列中继续填充剩余的 top-5
    while queue and len(top_k_results) < max_paths:
        current_score, current_path = heapq.heappop(queue)
        # print(f"Adding remaining path: {current_path} with score: {-current_score}")
        top_k_results.append((current_path, -current_score))

    # 按得分排序，返回前五条路径
    top_k_results.sort(key=lambda x: x[1], reverse=True)

    return top_k_results


# 用于计算编辑距离的函数（Levenshtein Distance）
def levenshtein_distance(a, b):
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def lookahead_get_entity_neighbors(entity):
    query = """
    MATCH (e:Entity)-[r]-(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type, n.name AS neighbor_entity
    """

    # 执行查询
    result = session.run(query, entity_name=entity)

    neighbors = []
    seen_neighbors = set()  # 存储已访问的邻居实体，防止重复

    for record in result:
        rel_type = record["relationship_type"]
        neighbor = record["neighbor_entity"]

        # 检查是否已经访问过该邻居实体
        if neighbor not in seen_neighbors:
            seen_neighbors.add(neighbor)
            neighbors.append((rel_type, neighbor))

    return neighbors

def find_all_shortest_paths(entities, max_hops=5):
    all_paths = {}
    for start_entity, end_entity in itertools.permutations(entities, 2):
        try:
            paths = find_shortest_path(start_entity, end_entity, max_hops)
            if paths:  # 如果找到了路径就加入
                all_paths[(start_entity, end_entity)] = paths
            else:
                print(f"Skipped: No path found between '{start_entity}' and '{end_entity}'")
        except Exception as e:
            print(f"Error finding path between '{start_entity}' and '{end_entity}': {e}")
            continue
    return all_paths


def is_contiguous_sublist(sub, full):
    """
    判断 sub 是否是 full 的连续子列表
    例如：sub = ['A','B','C']，full = ['X','A','B','C','Y'] 返回 True
    """
    n, m = len(full), len(sub)
    for i in range(n - m + 1):
        if full[i:i + m] == sub:
            return True
    return False

def remove_duplicate_entities(path):
    """去除路径中相邻重复的实体"""
    if isinstance(path, str):
        path_elements = path.split(" -> ")
    else:
        path_elements = path
    processed = []
    for item in path_elements:
        item = item.strip()
        if not processed or item != processed[-1]:
            processed.append(item)
    return processed

def combine_paths(entities, all_paths):
    """
    尽量串联出包含最多 match_kg 实体的路径
    如果在一个排列中某对连续实体没有路径，则将前面的连续段记录下来，
    最后对所有段进行过滤，去掉那些是其他段的连续子列表。
    如果没有任何段，则返回所有两两路径作为 fallback。
    """
    candidate_segments = []
    # 遍历所有排列
    for permutation in itertools.permutations(entities):
        current_segment = []  # 存放当前排列的连续拼接段
        for i in range(len(permutation) - 1):
            start_entity = permutation[i]
            end_entity = permutation[i + 1]
            if (start_entity, end_entity) in all_paths:
                # 找到该对的路径，取第一条最短路径
                path = all_paths[(start_entity, end_entity)][0]
                path_elements = path.split(" -> ")
                if not current_segment:
                    current_segment = path_elements
                else:
                    # 如果当前段的最后一个实体与新路径的起点一致，则合并（去掉重复起点）
                    if current_segment[-1] == path_elements[0]:
                        current_segment.extend(path_elements[1:])
                    else:
                        current_segment.extend(path_elements)
            else:
                # 如果当前对没有路径，则将当前连续段保存（如果非空），并重置
                if current_segment:
                    candidate_segments.append(remove_duplicate_entities(current_segment))
                    current_segment = []
        # 结束当前排列时，如果有非空段，保存下来
        if current_segment:
            candidate_segments.append(remove_duplicate_entities(current_segment))
    # 过滤掉长度太短的（只包含一个实体或没有关系的）
    candidate_segments = [seg for seg in candidate_segments if len(seg) >= 3]

    # 过滤掉那些是其他段的连续子列表
    maximal_segments = []
    for seg in candidate_segments:
        is_sub = False
        for other in candidate_segments:
            if seg != other and is_contiguous_sublist(seg, other):
                is_sub = True
                break
        if not is_sub:
            maximal_segments.append(seg)

    # 如果没有找到任何连续段，则采用 fallback：返回所有两两路径
    if not maximal_segments:
        print("⚠️ No contiguous segments found. Falling back to pairwise paths.")
        fallback_segments = []
        for key, paths in all_paths.items():
            for path in paths:
                fallback_segments.append(remove_duplicate_entities(path.split(" -> ")))
        return fallback_segments

    return maximal_segments


# 查找两个实体之间的最短路径的函数。它使用了Cypher查询语言来查找Neo4j图数据库中两个节点（实体）之间的最短路径，并返回路径的字符串表示
# paths = ["Entity A->RELATION 1->Entity B->RELATION 2->Entity C", "Entity X->RELATION 3->Entity Y"]
# exist_entity = "Entity B"
def find_shortest_path(start_entity_name, end_entity_name, max_hops=5):
    global driver
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            f"MATCH p = allShortestPaths((start_entity)-[*..{max_hops}]-(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        for record in result:
            path = record["p"]
            entities = []
            relations = []

            # 收集路径中节点和关系
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)

            # 打印路径的三元组信息
            # print(f"\n=== Path from '{start_entity_name}' to '{end_entity_name}' ===")
            # for i in range(len(relations)):
            #     e1 = entities[i].replace("_", " ")
            #     rel = relations[i].replace("_", " ")
            #     e2 = entities[i + 1].replace("_", " ")
            #     print(f"Triple: ({e1}) -[{rel}]-> ({e2})")

            # 构建路径字符串
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_", " ")
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_", " ")
                    path_str += " -> " + relations[i] + " -> "
            paths.append(path_str.strip(" -> "))

        return paths


def extract_features(query, paths):
    def levenshtein_distance(a, b):
        if len(a) < len(b):
            return levenshtein_distance(b, a)
        if len(b) == 0:
            return len(a)
        previous_row = range(len(b) + 1)
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def jaccard_similarity(a, b):
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union != 0 else 0

    features = []
    for path in paths:
        query_words = set(word_tokenize(query))
        path_words = set(word_tokenize(path))
        word_overlap = len(query_words & path_words)
        common_words = query_words & path_words

        edit_distance = levenshtein_distance(query, path)
        path_jumps = len(path.split(' -> ')) - 1
        path_length = len(path.split(' -> '))
        word_jaccard = jaccard_similarity(query_words, path_words)
        relation_count = path_jumps
        entity_count = path_length
        answer_count = path.count('answer')
        all_words_in_query = int(path_words.issubset(query_words))
        both_contain_numbers = int(any(char.isdigit() for char in query) and any(char.isdigit() for char in path))
        number_jaccard = jaccard_similarity(set(filter(str.isdigit, query)), set(filter(str.isdigit, path)))

        feature_row = [word_overlap, edit_distance, path_jumps, path_length,
                       word_jaccard, relation_count, entity_count, answer_count,
                       all_words_in_query, both_contain_numbers, number_jaccard]

        features.append((feature_row, common_words))
        # print(f"Path: {path}, Features: {feature_row}, Common Words: {common_words}")

    return pd.DataFrame([f[0] for f in features], columns=[
        'word_overlap', 'edit_distance', 'path_jumps', 'path_length',
        'word_jaccard', 'relation_count', 'entity_count', 'answer_count',
        'all_words_in_query', 'both_contain_numbers', 'number_jaccard'
    ])


def score_path(features_row):
    # 特征的权重
    scores = {
        'word_overlap': features_row['word_overlap'] * 2.0,  # 增加权重
        'edit_distance': (1 / (features_row['edit_distance'] + 1)) ** 0.5,  # 调整倒数的权重
        'path_jumps': -features_row['path_jumps'] * 0.5,  # 负得分，简洁的路径得分更高
        'path_length': -features_row['path_length'] * 0.5,  # 同样是负得分
        'word_jaccard': features_row['word_jaccard'] * 2.0,  # 增加权重
        'relation_count': features_row['relation_count'] * 1.0,
        'entity_count': features_row['entity_count'] * 1.0,
        'answer_count': features_row['answer_count'] * 1.0,
        'all_words_in_query': features_row['all_words_in_query'] * 1.5,
        'both_contain_numbers': features_row['both_contain_numbers'] * 1.0,
        'number_jaccard': features_row['number_jaccard'] * 1.0,

        # 新增的特征得分
        'possible_cure_disease': features_row.get('possible_cure_disease_count', 0) * 1.5,  # 假设权重为1.5
        'can_check_disease': features_row.get('can_check_disease_count', 0) * 2.0,  # 假设权重为1.5
        'need_medical_test': features_row.get('need_medical_test_count', 0) * 2.0,  # 假设权重为1.5
        'possible_disease': features_row.get('possible_disease_count', 0) * 1.5  # 假设权重为1.5
    }

    # 添加特征之间的相互作用
    interaction_score = features_row.get('word_overlap', 0) * features_row.get('word_jaccard', 0) * 0.5
    total_score = sum(scores.values()) + interaction_score

    #for feature, score in scores.items():
        #print(f"{feature} score: {score}")

#    print(f"Interaction score: {interaction_score}")
#     print(f"Total score: {total_score}")

    return total_score

def find_relevant_paths(question, result_paths):
    """
    根据问题 question 和给定的纯路径列表 result_paths，
    计算路径的特征、打分，最终选出得分最高的若干条。
    返回格式: [(processed_path_str, original_path_str, final_score), ...]
    """
    def process_path(path_str):
        """将 path_str 处理或转换为需要的格式"""
        # 如果 path_str 是列表/元组，就再拼接；如果本身是字符串，就直接返回
        if isinstance(path_str, list):
            return " -> ".join(str(x) for x in path_str)
        elif isinstance(path_str, str):
            return path_str
        else:
            return str(path_str)

    # 1. 对 question 和路径进行分词
    tokenized_query = word_tokenize(question)
    # 如果 result_paths 中每个元素本身就是字符串，则 tokenized_corpus 直接对它们分词即可
    # 若每个元素是 (relation, entity) 之类的列表，需要你先做 " -> ".join(...)
    paths_as_str = []
    for p in result_paths:
        if isinstance(p, list):
            p_str = " -> ".join(str(x) for x in p)
        else:
            p_str = str(p)
        paths_as_str.append(p_str)

    tokenized_corpus = [word_tokenize(p_str) for p_str in paths_as_str]

    if not tokenized_corpus:
        raise ValueError("Tokenized corpus is empty. Ensure result_paths are not empty.")

    # 2. 提取特征
    #   将“问题”转成字符串，把每条“路径”也转成字符串，交给 extract_features
    features_df = extract_features(" ".join(tokenized_query), paths_as_str)

    # 3. 计算每条路径的得分
    features_df["score"] = features_df.apply(score_path, axis=1)

    # ============ FIX: 这里必须是 zip(paths, features_df["score"]) ============
    #   否则就会把分数当做 path 解析！
    zipped = zip(paths_as_str, features_df["score"])
    # 按分数(即 x[1])降序排列
    sorted_paths = sorted(zipped, key=lambda x: x[1], reverse=True)
    top_k = sorted_paths[:20]

    # 4. 处理并返回
    processed_paths = []
    for path_str, final_score in top_k:
        # 先转成“可读”路径字符串
        processed_path = process_path(path_str)
        processed_paths.append((processed_path, path_str, final_score))

    return processed_paths

def get_entity_neighbors(entity_name: str, visited_entities: set) -> List[List[str]]:
    """获取实体的所有邻居，不考虑关系的方向"""
    query = """
    MATCH (e:Entity)-[r]-(n)  // 通过去掉关系的方向符号，将有向查询改为无向查询
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           n.name AS neighbor_entity
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    seen_paths = set()  # 用来存储和检查已经遇到的路径

    for record in result:
        rel_type = record["relationship_type"]
        neighbor = record["neighbor_entity"]

        # 确保 neighbor 是字符串类型
        neighbor = str(neighbor)

        # 如果邻居已经访问过，则跳过
        if neighbor in visited_entities:
            continue

        # 创建一个表示当前路径的字符串
        current_path = "{} -> {} -> {}".format(entity_name.replace("_", " "),
                                               rel_type.replace("_", " "),
                                               neighbor.replace("_", " "))

        # 检查路径是否已经存在
        if current_path not in seen_paths:
            seen_paths.add(current_path)
            neighbor_list.append([entity_name.replace("_", " "),
                                  rel_type.replace("_", " "),
                                  neighbor.replace("_", " ")])
            # 标记该邻居已访问
            visited_entities.add(neighbor)

    # 打印找到的邻居以进行调试
    if not neighbor_list:
        print(f"No neighbors found for entity: {entity_name}")
    else:
        print(f"Neighbors found for entity: {entity_name}: {neighbor_list}")

    return neighbor_list

# 计算两个向量之间余弦相似度
def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


def prompt_path_finding(path: str) -> str:
    template = """
    There are some knowledge graph paths. They follow the entity->relationship->entity->relationship->entity format.  
    \n\n
    {Path}
    \n\n
     Convert each path to a natural language statement, using single quotation marks for entity names and relation names. Name them as Evidence 1:..., Evidence 2:..., and so on. The natural language statement should be clear and concise.\n\n

    Output:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    system_message_prompt.format(Path=path)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path=path, \
                                                        text={})

    response_of_KG_path = chat(chat_prompt_with_values.to_messages(), model="gpt-3.5-turbo").content
    return response_of_KG_path

def final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor):
    """
    最终生成提示，结合路径证据与邻居证据，统一为单条 prompt 输入。
    """

    # ---------- 格式整理 ----------
    if isinstance(response_of_KG_list_path, list):
        response_of_KG_list_path = '\n'.join(response_of_KG_list_path)
    if isinstance(response_of_KG_neighbor, list):
        response_of_KG_neighbor = '\n'.join(response_of_KG_neighbor)

    # ---------- 构造单条 Prompt ----------
    prompt = f"""
You are a knowledgeable and reliable AI medical assistant. Based on the patient's description and the provided knowledge graph evidence, give a clear, concise answer that includes:

- Possible disease
- Recommended tests to confirm diagnosis
- Suggested medications or treatments

Do not include step-by-step reasoning or any Markdown formatting (e.g., **, #, ---). Use plain text with complete sentences.

Patient input:
{input_text}

Knowledge graph evidence:
{response_of_KG_list_path}

Additional related knowledge:
{response_of_KG_neighbor}

Here is an example of the expected answer format:

Sample Answer:
Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.

""".strip()

    # ---------- 调用模型 ----------
    response = chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content







if __name__ == "__main__":
    YOUR_OPENAI_KEY = ''
    os.environ['OPENAI_API_KEY'] = YOUR_OPENAI_KEY
    openai.api_key = YOUR_OPENAI_KEY
    os.environ['OPENAI_API_BASE'] = ''
    openai.api_base = 'https://api.openai-sb.com/v1'


    # neo4j数据库连接
    uri = "bolt://localhost:7687/"
    username = "neo4j"
    password = "neo4j"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    chat = ChatOpenAI(openai_api_key=YOUR_OPENAI_KEY)

    # 正则表达式设置
    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    with open('output/explainpe/label.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(
            ['Question', 'Label', 'MindMap', 'GPT3.5', 'BM25_retrieval', 'Embedding_retrieval', 'KG_retrieval', 'GPT4'])

    with open('./data/chatdoctor5k/entity_embeddings_MedBERT.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)

    with open('./data/chatdoctor5k/keyword_embeddings_MedBERT.pkl', 'rb') as f2:
        keyword_embeddings = pickle.load(f2)

    docs_dir = './data/chatdoctor5k/document'
    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r') as f:
            doc = f.read()
            docs.append(doc)

    # === 加载带有 Question 和 Navipath 的 CSV，只做一次 ===
    # navipath_df = pd.read_csv(r'evaluation\\chatdoctor5k\\output_NER_chatgpt_myall_add.csv',encoding='latin1')
    # navipath_df['Question_nospace'] = navipath_df['Question'].astype(str).str.replace(' ', '')
    with open("./data/chatdoctor5k/NER_chatgpt_choose.json", "r") as f:
        for line in f.readlines()[0:]:  # 根据需要调整起始行
            x = json.loads(line)
            input = x["qustion_output"]
            input = input.replace("\n", "").replace("<OOS>", "<EOS>").replace(":", "") + "<END>"
            input_text = re.findall(re3, input)
            if not input_text:
                continue
            print('Question:\n', input_text[0])
            print()


            question_kg = re.findall(re1, input)
            if not question_kg:
                question_kg = re.findall(re2, input)
                if not question_kg:
                    print("<Warning> no entities found", input)
                    continue
            question_kg = question_kg[0].replace("<END>", "").replace("<EOS>", "").replace("\n", "")
            question_kg = question_kg.split(", ")
            print("question_kg:", question_kg)
            print()

            # 提取答案
            output = x["answer_output"]
            output = output.replace("\n","")
            output = output.replace("<OOS>","<EOS>")
            output = output.replace(":","") + "<END>"
            output_text = re.findall(re3,output)

            match_kg =align_entities_to_kg_entities(question_kg,0.5)
            print('match_kg', match_kg)
            print()

            # Step 1: 使用 COT 扩展实体
            cot_candidates = cot_expand_entities_batch(input_text[0])
            print("COT Expanded Entities:", cot_candidates)

            cot_candidates = align_entities_to_kg_entities(cot_candidates,0.05)
            print("COT expanded Entities_kg:", cot_candidates)

            # Step 2: 对扩展实体进行打分（向量相似度 + 路径跳数）
            scored_expanded_entities = []
            for candidate in cot_candidates:
                if candidate in match_kg:
                    continue  # 避免重复

                try:
                    candidate_index = entity_embeddings["entities"].index(candidate.replace("_", " "))
                    candidate_emb = np.array(entity_embeddings["embeddings"][candidate_index])
                except ValueError:
                    continue  # 未匹配到候选实体

                total_score = 0
                valid_refs = 0

                for origin_entity in match_kg:
                    try:
                        origin_index = entity_embeddings["entities"].index(origin_entity.replace("_", " "))
                        origin_emb = np.array(entity_embeddings["embeddings"][origin_index])
                    except ValueError:
                        continue

                    sim = cosine_similarity_manual(candidate_emb.reshape(1, -1), origin_emb.reshape(1, -1))[0][0]
                    path_len = get_shortest_path_length(origin_entity, candidate,session)
                    if path_len is None:
                        continue

                    path_score = 1 / (path_len + 1)
                    score = sim + path_score
                    total_score += score
                    valid_refs += 1

                if valid_refs > 0:
                    avg_score = total_score / valid_refs
                    scored_expanded_entities.append((candidate, avg_score))

            # Step 3: 选出 top-k 扩展实体
            # 排序得分
            scored_expanded_entities = sorted(scored_expanded_entities, key=lambda x: x[1], reverse=True)

            # 根据 match_kg 的长度动态选择 top-k 数量
            num_match_entities = len(match_kg)

            if num_match_entities >= 7:
                expanded_entities = []  # 跳过，不选
            elif num_match_entities == 6:
                expanded_entities = [e for e, _ in scored_expanded_entities[:1]]
            elif num_match_entities == 5:
                expanded_entities = [e for e, _ in scored_expanded_entities[:2]]
            else:  # <= 4
                expanded_entities = [e for e, _ in scored_expanded_entities[:3]]

            print("Top Expanded Entities:", expanded_entities)


            # Step 4: 合并为最终实体集并去重
            final_kg = list(set(match_kg + expanded_entities))
            print("Final entity set for KG path finding:", final_kg)

            # Step 5: 利用 combine_paths 得到初步的候选路径
            all_paths = find_all_shortest_paths(final_kg)
            result_path = combine_paths(final_kg, all_paths)
            # 对路径字符串去重（无论是 list 还是 str，都处理为统一的 " -> " 形式）
            result_path_dedup = []
            seen = set()
            for p in result_path:
                if isinstance(p, list):
                    path_str = " -> ".join(p)
                else:
                    path_str = str(p)
                if path_str not in seen:
                    seen.add(path_str)
                    result_path_dedup.append(p)

            result_path = result_path_dedup  # 替换原来的路径
            print("Deduplicated combined paths:", result_path)
            print()

            # 如果路径为空则跳过该步骤，避免无用处理
            if not result_path:
                print("⚠️ Deduplicated combined paths is empty, skipping path processing.\n")

            representative_candidates = []  # 确保它在任何情况下都被定义

            # Step 6: 获取候选路径（三元组格式）
            # 如果路径为空则跳过该路径的处理，直接继续后面的步骤
            if not result_path:
                print("⚠️ Deduplicated combined paths is empty, skipping path processing.\n")
                # 直接进入后续步骤
            else:
                top_relevant_results = find_relevant_paths(input_text[0], result_path)
                # 提取候选路径和对应的得分
                top_candidate_paths = [item[0] for item in top_relevant_results]
                candidate_scores = [item[2] for item in top_relevant_results]

                print("Top candidate structured paths (before clustering):")
                for p in top_candidate_paths:
                    print("  " + p)
                print()

                # 对候选路径进行聚类，选出代表性路径（这里设定期望聚成5个簇）
                representative_candidates = cluster_candidates(top_candidate_paths, candidate_scores, num_clusters=4)

                print("Representative candidate paths after clustering:")
                for p in representative_candidates:
                    print("  " + p)
                print()

            # Step 7: 对实体集中的每个实体使用 lookahead 生成邻居证据
            neighbor_evidence = []
            for entity in final_kg:
                # 使用 lookahead 扩展，每个实体扩展一跳，返回最多 3 条候选扩展路径
                expanded_neighbors = top_k_paths_with_lookahead(
                    initial_entity=entity,
                    question=input_text[0],
                    max_depth=4,
                    max_lookahead=2,  # 轻量扩展，避免信息爆炸
                    max_paths=4
                )
                if expanded_neighbors:
                    # 选取得分最高的扩展结果
                    best_neighbor = max(expanded_neighbors, key=lambda x: x[1])
                    ext_path, ext_score = best_neighbor
                    # ext_path 的格式通常为 [(None, entity), (relation, neighbor), ...]
                    if isinstance(ext_path, list) and len(ext_path) > 1:
                        ext_entities = [item[1] for item in ext_path][1:]  # 去掉第一个重复实体
                        neighbor_path = entity + " -> " + " -> ".join(ext_entities)
                        neighbor_evidence.append(neighbor_path)
                    else:
                        neighbor_evidence.append(entity)
            # 去重，并选取前5条邻居证据
            neighbor_evidence = list(dict.fromkeys(neighbor_evidence))
            top_neighbor_evidence = neighbor_evidence[:5]

            print("Top neighbor evidence (lookahead-generated):")
            for p in top_neighbor_evidence:
                print("  " + p)
            print()


            # Step 8: 利用 prompt_path_finding 将结构化文本转换为自然语言描述
            if representative_candidates:  # 如果代表性候选路径不为空
                path_structured_text = "\n".join(representative_candidates)
                print("response_of_KG_list_path:\n", path_structured_text)
                print()
                response_of_KG_list_path = prompt_path_finding(path_structured_text)
                print("LLM converted natural language evidence:\n", response_of_KG_list_path)
                print()
            else:
                # 如果没有代表性路径，直接写空值
                response_of_KG_list_path = ""  # 设置为空字符串
                print("No representative candidates found, skipping path generation.\n")
                print("LLM converted natural language evidence:\n", response_of_KG_list_path)
                print()


            # 将邻居证据转为自然语言
            neighbor_structured_text = "\n".join(top_neighbor_evidence)
            response_of_KG_neighbor = prompt_path_finding(neighbor_structured_text)
            print("response_of_KG_neighbor:\n", neighbor_structured_text)
            print()
            print("LLM converted natural language neighbor evidence:\n", response_of_KG_neighbor)
            print()


            # 最终回答生成
            output_all = final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor)
            print('\nMy_Answer:\n', output_all)



