---
language: en
license: mit
tags:
- legal
- court-views-generation
- cvg
- nlp
- knowledge-graph
- legal-ai
- irac
- us-law
- criminal-law
---

# 🏛️ US Court Views Generation (CVG) Dataset & Legal Fact-Rule Knowledge Graph

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ACL%20Findings%202024-blue)](https://aclanthology.org/2024.findings-acl.918/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.03600-b31b1b.svg)](https://arxiv.org/abs/2406.03600)
[![Dataset](https://img.shields.io/badge/Dataset-923K%20Cases-green)](https://github.com/YANGWU001/US_CVG_dataset)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A Large-Scale English Legal Dataset for Court Views Generation and Legal AI Research*

**[Yang Wu](https://github.com/YANGWU001)** · **Chenghao Wang** · **Ece Gumusel** · **Xiaozhong Liu**

Worcester Polytechric Institute · Peking University · Indiana University at Bloomington

</div>

---

## 📖 Overview

This repository contains the **US Court Views Generation (CVG) Dataset** and **US Legal Fact-Rule Knowledge Graph**, introduced in our ACL Findings 2024 paper: [*"Knowledge-Infused Legal Wisdom: Navigating LLM Consultation through the Lens of Diagnostics and Positive-Unlabeled Reinforcement Learning"*](https://aclanthology.org/2024.findings-acl.918/).

### What is Court Views Generation (CVG)?

Court Views Generation is the task of predicting judicial decisions and generating court opinions based on case descriptions. Our dataset provides a large-scale, structured resource for:

- 🏛️ **Legal AI Research**: Training and evaluating LLMs on legal reasoning tasks
- 📊 **Judicial Decision Prediction**: Predicting court outcomes from case facts
- 🔍 **Legal Information Retrieval**: Understanding relationships between legal facts and rules
- 💡 **Legal Assistant Systems**: Building AI-powered legal consultation tools
- 📚 **Legal NLP**: Advancing natural language processing in the legal domain

---

## 🎯 Key Features

### 1. Court Views Generation (CVG) Dataset

- **923,127 US criminal cases** structured in IRAC format
- **Source**: [Caselaw Access Project (CAP)](https://case.law/) - Harvard Law School
- **Format**: IRAC (Issue, Rule, Analysis, Conclusion)
- **Domain**: US Criminal Law
- **Language**: English
- **Size**: ~5 compressed files (gzipped)

#### IRAC Structure

Each case is formatted according to the legal reasoning framework:

| Component | Description | Usage in CVG |
|-----------|-------------|--------------|
| **Issue (I)** | The legal question(s) to be addressed | Part of fact description |
| **Rule (R)** | Applicable legal statutes, regulations, case law | Part of fact description |
| **Analysis (A)** | Application of rules to the facts | Part of fact description |
| **Conclusion (C)** | Court's judgment on the legal issue | **Target (Court Views)** |

> **Note**: IRA (Issue + Rule + Analysis) serves as the **input fact description**, while C (Conclusion) serves as the **target court views** for generation tasks.

### 2. US Legal Fact-Rule Knowledge Graph

- **Built from**: 11,000 carefully curated US criminal cases
- **Node Types**: Facts and Rules
- **Relationship Types**: 
  - `Depends On` (Fact → Fact)
  - `Complies With` (Fact → Rule)
  - `Violates` (Fact → Rule)
- **Pruning**: Nodes appearing < 10 times removed for clarity
- **Format**: NetworkX gpickle + CSV node list

#### Knowledge Graph Benefits

✅ **Causal Reasoning**: Understand dependencies between legal facts  
✅ **Chain-of-Thought**: Support multi-step legal reasoning  
✅ **Fact Completion**: Identify missing critical information in case descriptions  
✅ **Legal Analytics**: Analyze patterns in legal decision-making  
✅ **Interpretability**: Provide explainable AI in legal contexts

---

## 📊 Dataset Statistics

### CVG Dataset

| Metric | Value |
|--------|-------|
| **Total Cases** | 923,127 |
| **Domain** | US Criminal Law |
| **Average Case Length** | ~1,200 tokens |
| **Compressed Size** | ~X GB (5 files) |
| **Decompressed Size** | ~X GB |
| **Data Source** | CAP (case.law) |
| **Coverage** | Federal & State Courts |

### Knowledge Graph

| Metric | Value |
|--------|-------|
| **Source Cases** | 11,000 |
| **Total Nodes** | X (Facts) + Y (Rules) |
| **Total Edges** | Z |
| **Edge Types** | 3 (Depends On, Complies With, Violates) |
| **Min Node Frequency** | 10 occurrences |
| **Graph Type** | Directed Bipartite |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YANGWU001/US_CVG_dataset.git
cd US_CVG_dataset

# Install dependencies
pip install pandas networkx
```

### Usage: CVG Dataset

```python
import json
import gzip

# Load compressed IRAC data
def load_cvg_data(file_index=1):
    """
    Load Court Views Generation dataset from compressed files
    
    Args:
        file_index: Index of the file (1-5)
    
    Returns:
        List of case dictionaries with IRAC structure
    """
    file_path = f'CVG_dataset/all_compressed_irac_{file_index}.gz'
    
    with gzip.open(file_path, 'rb') as f:
        decompressed_data = f.read()
    
    data_as_str = decompressed_data.decode('utf-8')
    cases = data_as_str.split('|||||')
    
    return cases

# Example: Load first file
cases = load_cvg_data(file_index=1)
print(f"Loaded {len(cases)} cases")

# Example: Parse a single case
sample_case = cases[0]
print("Sample Case:")
print(sample_case)
```

### Usage: Fact-Rule Knowledge Graph

```python
import pickle
import pandas as pd
import networkx as nx

# Load the knowledge graph
def load_knowledge_graph():
    """
    Load US Legal Fact-Rule Knowledge Graph
    
    Returns:
        G: NetworkX graph object
        fact_nodes: List of fact node IDs
        rule_nodes: List of rule node IDs
        all_nodes: DataFrame with node metadata
    """
    # Load graph structure
    graph_path = 'fact_rule_KG/final_graph.gpickle'
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    # Load node metadata
    nodes_df = pd.read_csv('fact_rule_KG/final_nd.csv')
    
    # Separate fact and rule nodes (bipartite graph)
    fact_nodes = nodes_df[nodes_df["bipartite"] == 0].node.tolist()
    rule_nodes = nodes_df[nodes_df["bipartite"] == 1].node.tolist()
    all_nodes = nodes_df.node.tolist()
    
    return G, fact_nodes, rule_nodes, nodes_df

# Example: Load and explore the graph
G, fact_nodes, rule_nodes, nodes_df = load_knowledge_graph()

print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")
print(f"Fact Nodes: {len(fact_nodes)}")
print(f"Rule Nodes: {len(rule_nodes)}")

# Example: Get neighbors of a fact node
sample_fact = fact_nodes[0]
neighbors = list(G.neighbors(sample_fact))
print(f"\nNeighbors of '{sample_fact}': {neighbors}")

# Example: Find all "Violates" relationships
violates_edges = [
    (u, v) for u, v, data in G.edges(data=True) 
    if data.get('relation') == 'Violates'
]
print(f"\n'Violates' relationships: {len(violates_edges)}")
```

### Advanced Usage: Fact Completion Task

```python
def find_missing_facts(case_facts, knowledge_graph, fact_nodes):
    """
    Identify potentially missing facts in a case description
    using the knowledge graph
    
    Args:
        case_facts: List of facts mentioned in the case
        knowledge_graph: NetworkX graph
        fact_nodes: List of all fact node IDs
    
    Returns:
        List of potentially missing facts
    """
    mentioned_facts = set(case_facts)
    candidate_missing = set()
    
    # For each mentioned fact, check its dependencies
    for fact in mentioned_facts:
        if fact in knowledge_graph:
            # Get facts that this fact depends on
            predecessors = set(knowledge_graph.predecessors(fact))
            # Find facts not mentioned
            missing = predecessors - mentioned_facts
            candidate_missing.update(missing)
    
    return list(candidate_missing)

# Example usage
case_facts = ["Assault", "Intent", "Physical Contact"]
missing = find_missing_facts(case_facts, G, fact_nodes)
print(f"Potentially missing facts: {missing}")
```

---

## 📚 Applications

### 1. **Legal Large Language Models (LLMs)**

Train and fine-tune LLMs for legal tasks:

```python
# Example: Prepare training data for LLM fine-tuning
def prepare_cvg_training_data(cases):
    """
    Format CVG dataset for LLM training
    """
    training_examples = []
    
    for case in cases:
        # Parse IRAC components (implementation depends on data format)
        issue, rule, analysis, conclusion = parse_irac(case)
        
        # Create input-output pairs
        input_text = f"Issue: {issue}\nRule: {rule}\nAnalysis: {analysis}\n\nGenerate the court's conclusion:"
        output_text = conclusion
        
        training_examples.append({
            "input": input_text,
            "output": output_text
        })
    
    return training_examples
```

### 2. **Diagnostic Legal Assistant (D3LM)**

Build interactive legal consultation systems:

- Use knowledge graph to identify missing critical facts
- Generate diagnostic questions to elicit more information
- Provide accurate court view predictions

### 3. **Legal Information Retrieval**

```python
def retrieve_similar_cases(query_facts, knowledge_graph):
    """
    Find cases with similar fact patterns using the knowledge graph
    """
    # Expand query with related facts from KG
    expanded_facts = set(query_facts)
    
    for fact in query_facts:
        if fact in knowledge_graph:
            neighbors = knowledge_graph.neighbors(fact)
            expanded_facts.update(neighbors)
    
    # Search for cases mentioning expanded facts
    # (Implementation depends on indexing strategy)
    return expanded_facts
```

### 4. **Positive-Unlabeled Reinforcement Learning (PURL)**

Implement our PURL algorithm for adaptive question generation:

1. Extract case subgraph from knowledge graph
2. Identify critical missing facts (positive nodes)
3. Use RL to learn optimal question generation policy
4. Iteratively refine court view predictions

---

## 🛠️ Data Format

### CVG Dataset Format

Each case is stored as a plain text block with IRAC sections (format may vary):

```
Case ID: [case_id]
Issue: [Legal question to be addressed]
Rule: [Applicable laws and precedents]
Analysis: [Application of rules to facts]
Conclusion: [Court's decision and reasoning]
```

Cases are separated by `|||||` delimiter.

### Knowledge Graph Format

**Node CSV (`final_nd.csv`)**:

```csv
node,bipartite,frequency
"Assault",0,450
"Intent",0,380
"Criminal Law",1,1200
```

- `node`: Node name (fact or rule)
- `bipartite`: 0 for facts, 1 for rules
- `frequency`: Number of occurrences in source cases

**Graph Pickle (`final_graph.gpickle`)**:

NetworkX directed graph with edge attributes:

```python
{
    ('Assault', 'Intent'): {'relation': 'Depends On'},
    ('Intent', 'Criminal Law'): {'relation': 'Complies With'},
    ('Excessive Force', 'Use of Force Rules'): {'relation': 'Violates'}
}
```

---

## 📄 Citation

If you use this dataset in your research, please cite our paper:

### BibTeX

```bibtex
@article{wu2024knowledge,
  title={Knowledge-infused legal wisdom: Navigating llm consultation through the lens of diagnostics and positive-unlabeled reinforcement learning},
  author={Wu, Yang and Wang, Chenghao and Gumusel, Ece and Liu, Xiaozhong},
  journal={arXiv preprint arXiv:2406.03600},
  year={2024}
}
```

### ACL Anthology

```bibtex
@inproceedings{wu-etal-2024-knowledge,
    title = "Knowledge-Infused Legal Wisdom: Navigating {LLM} Consultation through the Lens of Diagnostics and Positive-Unlabeled Reinforcement Learning",
    author = "Wu, Yang and Wang, Chenghao and Gumusel, Ece and Liu, Xiaozhong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.918/",
    pages = "15447--15463"
}
```

---

## 🔗 Related Resources

### Paper & Code

- **Paper (ACL Anthology)**: https://aclanthology.org/2024.findings-acl.918/
- **Paper (arXiv)**: https://arxiv.org/abs/2406.03600
- **Main Repository**: https://github.com/YANGWU001/US_CVG_dataset

### Related Datasets

- **CAP (Caselaw Access Project)**: https://case.law/
- **LEDGAR**: Legal Entity Detection & Relation Classification
- **CaseHOLD**: Legal Citation Prediction
- **LegalBench**: Legal Reasoning Benchmark

### Related Work

- **LegalBERT**: BERT for Legal Text
- **LEGAL-BERT**: Domain-Adapted BERT
- **InCaseLaw**: Indian Case Law Dataset
- **COLIEE**: Competition on Legal Information Extraction/Entailment

---

## 📋 Data License & Ethics

### License

This dataset is released under the **MIT License**.

- ✅ **Free to use** for research and commercial purposes
- ✅ **Modification and distribution** permitted
- ⚠️ **Attribution required**: Please cite our paper

### Data Source

All cases are sourced from the [Caselaw Access Project (CAP)](https://case.law/), which provides free public access to US case law. CAP is a collaboration between Harvard Law School Library Innovation Lab and Harvard University Library.

### Ethical Considerations

⚠️ **Important Disclaimers**:

1. **Not Legal Advice**: This dataset is for research purposes only and does not constitute legal advice
2. **Historical Data**: Cases reflect historical legal decisions that may not represent current law
3. **Bias Awareness**: Legal data may contain historical biases; users should be aware of potential fairness issues
4. **Privacy**: All data is from public court records; personal identifying information should be handled responsibly
5. **Professional Consultation**: Any real-world legal application should involve qualified legal professionals

### Intended Use

✅ **Appropriate Uses**:
- Academic research in legal AI and NLP
- Training and evaluating legal language models
- Legal information retrieval systems
- Educational tools for legal reasoning
- Legal analytics and empirical legal studies

❌ **Inappropriate Uses**:
- Direct legal advice to individuals without lawyer oversight
- Automated decision-making in legal proceedings without human review
- Systems that could disadvantage protected groups

---

## 🤝 Contributing

We welcome contributions to improve this dataset and knowledge graph!

### Ways to Contribute

- **Data Quality**: Report errors or inconsistencies in the dataset
- **Additional Annotations**: Contribute new annotations (e.g., case outcomes, jurisdictions)
- **Tools**: Develop tools for easier data access and analysis
- **Benchmarks**: Create evaluation benchmarks using this dataset
- **Documentation**: Improve documentation and examples

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a Pull Request

---

## 📊 Benchmarks & Leaderboards

We encourage researchers to use this dataset for benchmarking. If you publish results, please share them with us so we can maintain a leaderboard!

### Evaluation Metrics

For **Court Views Generation (CVG)**:

- **ROUGE-L**: Longest common subsequence matching
- **BLEU**: N-gram precision
- **BERTScore**: Semantic similarity using BERT embeddings
- **Legal-Specific Metrics**: Accuracy of legal reasoning steps

### Baseline Results

| Model | ROUGE-L | BLEU-4 | BERTScore | Notes |
|-------|---------|--------|-----------|-------|
| GPT-3.5 | TBD | TBD | TBD | Zero-shot |
| D3LM | TBD | TBD | TBD | With KG (from paper) |
| Your Model | - | - | - | Submit your results! |

*Submit your results via GitHub Issues or Pull Request*

---

## 🔧 Technical Details

### System Requirements

- **Storage**: ~X GB for full dataset
- **RAM**: 16GB+ recommended for loading full knowledge graph
- **Python**: 3.8+
- **Dependencies**: `pandas`, `networkx`, `numpy`

### Dataset Splits

For reproducible experiments, we recommend:

- **Training**: 80% of cases (~738K cases)
- **Validation**: 10% of cases (~92K cases)
- **Test**: 10% of cases (~92K cases)

### Known Issues & Limitations

1. **Format Variability**: Case formatting may vary due to OCR and parsing
2. **Incomplete Cases**: Some cases may have missing IRAC components
3. **Domain Coverage**: Dataset focuses on criminal law; civil cases not included
4. **Jurisdiction**: Mix of federal and state cases with varying legal standards
5. **Knowledge Graph Coverage**: KG built from subset (11K cases) of full dataset

---

## 📞 Contact

### Authors

- **Yang Wu** - Worcester Polytechnic Institute - [ywu19@wpi.edu](mailto:ywu19@wpi.edu)
- **Chenghao Wang** - Peking University
- **Ece Gumusel** - Indiana University at Bloomington
- **Xiaozhong Liu** - Worcester Polytechnic Institute - [xliu14@wpi.edu](mailto:xliu14@wpi.edu)

### Support

- **Issues**: [GitHub Issues](https://github.com/YANGWU001/US_CVG_dataset/issues)
- **Questions**: Open a discussion on GitHub
- **Collaboration**: Email the corresponding authors

---

## 🙏 Acknowledgments

We thank:

- **Harvard Law School Library Innovation Lab** for the Caselaw Access Project (CAP)
- **ACL Findings 2024** reviewers for their valuable feedback
- The **legal NLP research community** for inspiring this work
- Our institutions: **WPI**, **Peking University**, and **Indiana University**

### Funding

*Add funding acknowledgments if applicable*

---

## 📝 Changelog

### Version 1.0.0 (2024)

- ✅ Initial release of CVG dataset (923K cases)
- ✅ US Legal Fact-Rule Knowledge Graph
- ✅ Comprehensive documentation and examples
- ✅ ACL Findings 2024 publication

### Future Releases

- [ ] **v1.1**: Add jurisdiction and court type annotations
- [ ] **v1.2**: Include case outcomes (affirmed/reversed/remanded)
- [ ] **v2.0**: Expand to civil law cases
- [ ] **v2.1**: Multi-lingual support (starting with Spanish)

---

## 🌟 References

[1] **Howard Gensler**. 1985. IRAC: One more time. *Duquesne Law Review*, 24:243.

[2] **Caselaw Access Project**. 2024. Caselaw access project data. Online. Accessed: 2024. https://case.law/

[3] **Yang Wu, Chenghao Wang, Ece Gumusel, Xiaozhong Liu**. 2024. Knowledge-Infused Legal Wisdom: Navigating LLM Consultation through the Lens of Diagnostics and Positive-Unlabeled Reinforcement Learning. In *Findings of ACL 2024*.

---

<div align="center">

**⭐ If you find this dataset useful, please star the repository! ⭐**

**Made with ❤️ for the Legal AI Research Community**

---

[📄 Paper](https://aclanthology.org/2024.findings-acl.918/) • [🔗 Dataset](https://github.com/YANGWU001/US_CVG_dataset) • [📧 Contact](mailto:ywu19@wpi.edu) • [🐛 Issues](https://github.com/YANGWU001/US_CVG_dataset/issues)

---

*Last Updated: January 2025*

</div>
