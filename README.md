# [Knowledge-Infused Legal Wisdom: Navigating LLM Consultation through the Lens of Diagnostics and Positive-Unlabeled Reinforcement Learning]
[ACL Findings 2024] Yang Wu, Chenghao Wang, Ece Gumusel and Xiaozhong Liu
## Overview
This repository is the source of US court views generation (CVG) dataset. There are total 923127 criminal cases structured into IRAC (Issue, Rules, Analysis, Conclusion) [1] format. The raw data source is from US caselaw data CAP[2]. IRA can be used as fact description, and C(Conclusion) can be used as court views in practice.

If you find this dataset is useful for your research, please cite us.

## How to use
```setup
import json
import gzip
with gzip.open('all_compressed_irac_i.gz', 'rb') as f:
    decompressed_data = f.read()
data_as_str = decompressed_data.decode('utf-8')
cases = data_as_str.split('|||||')

```

## References
[1] Howard Gensler. 1985. Irac: One more time. Duq. L. Rev., 24:243.


[2] Caselaw Access Project. 2024. Caselaw access project data. Online. Accessed: 2024. https://case.law/

