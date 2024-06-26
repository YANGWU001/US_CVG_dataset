# US Legal Court Views Generation Dataset and Fact Rule Knowledge Graph
Knowledge-Infused Legal Wisdom: Navigating LLM Consultation through the Lens of Diagnostics and Positive-Unlabeled Reinforcement Learning.
[ACL Findings 2024] Yang Wu, Chenghao Wang, Ece Gumusel and Xiaozhong Liu
## Overview
This repository is the source of US court views generation (CVG) dataset in our [paper](https://arxiv.org/abs/2406.03600). There are total 923127 criminal cases structured into IRAC (Issue, Rules, Analysis, Conclusion) [1] format. The raw data source is from US caselaw data CAP[2]. IRA can be used as fact description, and C(Conclusion) can be used as court views in practice.

We have also developed a comprehensive US Legal Fact-Rule Knowledge Graph, detailed in our publication, which is built upon an extensive dataset of 11,000 US criminal cases. To enhance the clarity and utility of the graph, we have pruned nodes that appear less than 10 times across the dataset. This knowledge graph features two types of nodes (facts and rules), and establishes three kinds of directed relationships: "depends on" between facts, "complies with" between facts and rules, and "violates" also between facts and rules.

This structured approach not only facilitates a clearer understanding of the legal proceedings but also significantly contributes to causal reasoning and the development of chain-of-thought processes in legal analytics. This knowledge graph aims to provide a foundational tool for researchers and legal professionals to navigate through the complexities of criminal cases with enhanced accuracy and insight.

If you find this dataset is useful for your research, please cite us.

## How to use (CVG dataset)
```setup
import json
import gzip
with gzip.open('all_compressed_irac_i.gz', 'rb') as f:
    decompressed_data = f.read()
data_as_str = decompressed_data.decode('utf-8')
cases = data_as_str.split('|||||')

```


## How to use (Fact-rule knowledge graph)
```setup
import pickle
import pandas as pd
file_path = './final_graph.gpickle'
with open(file_path, 'rb') as f:
    final_graph = pickle.load(f)
final_nd = pd.read_csv("./final_nd.csv")
fact_nodes = final_nd[final_nd["bipartite"]==0].node.tolist()
all_nodes = final_nd.node.tolist()

```
## References
[1] Howard Gensler. 1985. Irac: One more time. Duq. L. Rev., 24:243.


[2] Caselaw Access Project. 2024. Caselaw access project data. Online. Accessed: 2024. https://case.law/

