from neo4j import GraphDatabase, basic_auth
import pandas as pd

# 1. build neo4j knowledge graph datasets
uri = "bolt://localhost:7687/"
username = "neo4j"
password = "neo4j"

driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()

##############################build KG

#session.run("MATCH (n) DETACH DELETE n")  # clean all

# read triples
df = pd.read_csv('train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])

for index, row in df.iterrows():
    head_name = row['head']
    tail_name = row['tail']
    relation_name = row['relation']

    query = (
            "MERGE (h:Entity { name: $head_name }) "
            "MERGE (t:Entity { name: $tail_name }) "
            "MERGE (h)-[r:`" + relation_name + "`]->(t)"
    )
    session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)