import os
import json
from py2neo import Graph, Node

class MedicalGraph:
    def __init__(self, start_record=0):
        # 初始化方法，设置数据文件路径和连接图数据库
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'medical.json')
        self.start_record = start_record  # 起始记录数
        self.g = Graph(
            "bolt://127.0.0.1:7687",
            user="neo4j",  # 数据库user name，如果没有更改过，应该是neo4j
            password="xyh888888")

    def read_cause(self):
        # 读取 JSON 文件并解析数据，提取Cause字段
        rels_cause = []  # 疾病与原因的关系

        count = 0
        with open(self.data_path, encoding='utf-8') as file:  # 指定编码为utf-8
            for data in file:
                count += 1
                if count < self.start_record:
                    continue  # 跳过已处理的数据
                data_json = json.loads(data)  # 解析 JSON 数据
                disease = data_json['name']

                if 'cause' in data_json:
                    cause = data_json['cause']
                    rels_cause.append([disease, cause.strip()])  # 移除前后空格

        return rels_cause

    def create_cause_relationships(self, rels_cause):
        # 创建疾病与原因之间的关系
        count = 0
        set_edges = []
        for edge in rels_cause:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "MERGE (p:Disease {name: '%s'}) MERGE (q:Cause {name: '%s'}) CREATE (p)-[rel:cause {name: '病因'}]->(q)" % (p, q)
            try:
                self.g.run(query)
                count += 1
                print("cause", count, all)
            except Exception as e:
                print(e)
        return

if __name__ == '__main__':
    start_record = 0  # 指定从哪个记录开始
    handler = MedicalGraph(start_record=start_record)
    print("step1:处理Cause关系")
    rels_cause = handler.read_cause()
    handler.create_cause_relationships(rels_cause)
