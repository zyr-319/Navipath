import os
import json
from py2neo import Graph, Node

class MedicalGraph:
    def __init__(self):
        # 初始化方法，设置数据文件路径和连接图数据库
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'medical.json')
        self.g = Graph(
            "bolt://127.0.0.1:7687",
            user="neo4j",  # 数据库user name，如果没有更改过，应该是neo4j
            password="xyh888888")

    def read_nodes(self):
        # 读取 JSON 文件并解析数据，提取所需字段，建立关系
        drugs = []
        checks = []
        diseases = []
        symptoms = []

        disease_infos = []

        rels_symptom = []  # 疾病与症状的关系
        rels_acompany = []  # 疾病与并发症的关系
        rels_check = []  # 疾病与检查项目的关系
        rels_recommanddrug = []  # 疾病与推荐药物的关系
        rels_prevent = []  # 疾病与预防措施的关系
        rels_get_way = []  # 疾病与传播途径的关系
        rels_cure_way = []  # 疾病与治疗方法的关系

        count = 0
        with open(self.data_path, encoding='utf-8') as file:  # 指定编码为utf-8
            for data in file:
                disease_dict = {}
                count += 1
                print(count)
                data_json = json.loads(data)  # 解析 JSON 数据
                disease = data_json['name']
                disease_dict['name'] = disease
                diseases.append(disease)

                # 提取各字段，如果存在则添加到字典中
                if 'prevent' in data_json:
                    disease_dict['prevent'] = data_json['prevent']
                    rels_prevent.append([disease, data_json['prevent']])
                if 'symptom' in data_json:
                    disease_dict['symptom'] = data_json['symptom']
                    symptoms += data_json['symptom']
                    for symptom in data_json['symptom']:
                        rels_symptom.append([disease, symptom])
                if 'get_way' in data_json:
                    disease_dict['get_way'] = data_json['get_way']
                    rels_get_way.append([disease, data_json['get_way']])
                if 'acompany' in data_json:
                    disease_dict['acompany'] = data_json['acompany']
                    for acompany in data_json['acompany']:
                        rels_acompany.append([disease, acompany])
                if 'cure_way' in data_json:
                    disease_dict['cure_way'] = data_json['cure_way']
                    rels_cure_way.append([disease, data_json['cure_way']])
                if 'check' in data_json:
                    disease_dict['check'] = data_json['check']
                    checks += data_json['check']
                    for _check in data_json['check']:
                        rels_check.append([disease, _check])
                if 'recommand_drug' in data_json:
                    disease_dict['recommand_drug'] = data_json['recommand_drug']
                    drugs += data_json['recommand_drug']
                    for drug in data_json['recommand_drug']:
                        rels_recommanddrug.append([disease, drug])

                disease_infos.append(disease_dict)

        return set(drugs), set(checks), set(symptoms), set(diseases), disease_infos, \
               rels_check, rels_recommanddrug, rels_symptom, rels_acompany, rels_prevent, rels_get_way, rels_cure_way

    def create_node(self, label, nodes):
        # 创建节点并添加Entity标签
        count = 0
        for node_name in nodes:
            node = Node(label, "Entity", name=node_name)
            self.g.create(node)
            count += 1
            print(count, len(nodes))
        return

    def create_diseases_nodes(self, disease_infos):
        # 创建疾病节点，包含指定的字段
        count = 0
        for disease_dict in disease_infos:
            node = Node("Disease", "Entity", name=disease_dict['name'],
                        prevent=disease_dict.get('prevent', ''),
                        symptom=disease_dict.get('symptom', []),
                        get_way=disease_dict.get('get_way', ''),
                        acompany=disease_dict.get('acompany', []),
                        cure_way=disease_dict.get('cure_way', []),
                        check=disease_dict.get('check', []),
                        recommand_drug=disease_dict.get('recommand_drug', []))
            self.g.create(node)
            count += 1
            print(count)
        return

    def create_graphnodes(self):
        # 创建所有图谱节点
        Drugs, Checks, Symptoms, Diseases, disease_infos, rels_check, rels_recommanddrug, rels_symptom, rels_acompany, rels_prevent, rels_get_way, rels_cure_way = self.read_nodes()
        self.create_diseases_nodes(disease_infos)
        self.create_node('Drug', Drugs)
        print(len(Drugs))
        self.create_node('Check', Checks)
        print(len(Checks))
        self.create_node('Symptom', Symptoms)
        return

    def create_graphrels(self):
        # 创建图谱中的关系边
        Drugs, Checks, Symptoms, Diseases, disease_infos, rels_check, rels_recommanddrug, rels_symptom, rels_acompany, rels_prevent, rels_get_way, rels_cure_way = self.read_nodes()
        self.create_relationship('Disease', 'Symptom', rels_symptom, 'has_symptom', '症状')
        self.create_relationship('Disease', 'Disease', rels_acompany, 'acompany_with', '并发症')
        self.create_relationship('Disease', 'Check', rels_check, 'need_check', '诊断检查')
        self.create_relationship('Disease', 'Drug', rels_recommanddrug, 'recommand_drug', '好评药品')
        self.create_relationship('Disease', 'Prevent', rels_prevent, 'prevent', '预防措施')
        self.create_relationship('Disease', 'GetWay', rels_get_way, 'get_way', '传播途径')
        self.create_relationship('Disease', 'CureWay', rels_cure_way, 'cure_way', '治疗方法')

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        # 创建实体之间的关系
        count = 0
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return

if __name__ == '__main__':
    handler = MedicalGraph()
    print("step1:导入图谱节点中")
    handler.create_graphnodes()
    print("step2:导入图谱边中")
    handler.create_graphrels()
