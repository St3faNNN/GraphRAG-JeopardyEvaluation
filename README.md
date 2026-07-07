# GraphRAG Jeopardy Evaluation

This repository evaluates and compares different Retrieval-Augmented Generation (RAG) approaches for generating Jeopardy-style quiz questions from answer entities. The project compares traditional dense-retrieval RAG models (RAG-Token and RAG-Sequence) against a structured GraphRAG approach built from knowledge graphs, with and without coreference resolution.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Core Concepts](#core-concepts)
  - [RAG-Token vs. RAG-Sequence](#rag-token-vs-rag-sequence)
  - [GraphRAG](#graphrag)
  - [Coreference Resolution in Knowledge Graphs](#coreference-resolution-in-knowledge-graphs)
- [Datasets](#datasets)
- [Models & Pipelines](#models--pipelines)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Overview

The goal is to generate high-quality Jeopardy clues (questions) given an answer entity (e.g., "Albert Einstein" → "In 1922 he was awarded the Nobel Prize in Physics"). We compare three paradigms:

1. **RAG-Token** – token-level retrieval-augmented generation using Wikipedia passages.
2. **RAG-Sequence** – sequence-level retrieval-augmented generation using Wikipedia passages.
3. **GraphRAG** – retrieval-augmented generation using structured knowledge-graph subgraphs.

The GraphRAG branch further splits into two variants to study the impact of entity disambiguation:

- **GraphRAG (no coreference)** – entities and pronouns remain as separate nodes.
- **GraphRAG (with coreference)** – aliases and pronouns are merged via coreference resolution.

---

## Folder Structure

```
GraphRAG-JeopardyEvaluation/
├── rag token/                     # RAG-Token pipeline & fine-tuned checkpoints
│   ├── rag_token.ipynb            # Training & inference notebook
│   └── rag_token_final/           # Saved model weights
│
├── rag sequence/                  # RAG-Sequence pipeline & fine-tuned checkpoints
│   ├── rag_seq.ipynb              # Training & inference notebook
│   └── rag_sequence_final/        # Saved model weights
│
├── knowledge graph/               # GraphRAG construction & evaluation
│   ├── kg_without_coref.ipynb     # KG builder (NO coreference resolution)
│   ├── kg_with_coref.ipynb        # KG builder (WITH coreference resolution)
│   ├── bart_eval_no_coref.ipynb   # BART eval on no-coref graph
│   ├── bart_eval_with_coref.ipynb # BART eval on with-coref graph
│   ├── knowledge_graph_no_coref.* # Saved graph files (110k nodes, 96k edges)
│   ├── knowledge_graph_with_coref.* # Saved graph files (122k nodes, 107k edges)
│   └── preparation/               # Data-splitting utilities
│
└── Answerability-Metric/          # Q-BLEU evaluation suite
    ├── answerability_score.py     # Main metric script
    ├── README.md
    └── ...
```

---

## Core Concepts

### RAG-Token vs. RAG-Sequence

Both are implementations of **Retrieval-Augmented Generation (RAG)** from Lewis et al. (2020). They combine a dense retriever (DPR) with a seq2seq generator (BART). The key difference lies in **how retrieved documents interact with the generation process**.

| Aspect | **RAG-Token** | **RAG-Sequence** |
|---|---|---|
| **Retrieval Granularity** | Each *token* can be generated conditioned on a *different* retrieved document. | The *entire sequence* is generated conditioned on a *single* retrieved document. |
| **Decoder Behavior** | Marginalizes over documents per token; more flexible but computationally heavier. | Marginalizes over documents once for the full sequence; more coherent. |
| **Use Case** | Better when different tokens need facts from different sources. | Better when a single consistent context is preferred. |
| **Model Class** | `RagTokenForGeneration` | `RagSequenceForGeneration` |
| **Pretrained Checkpoint** | `facebook/rag-token-nq` | `facebook/rag-sequence-nq` |

In this project, both models are fine-tuned on the SearchQA dataset (80k train samples) for 2 epochs with a frozen question encoder and a learning rate of `1e-5`.

> **Intuition:** RAG-Token is like a student who flips through multiple books for every word they write. RAG-Sequence picks one book and writes the whole sentence from it.

---

### GraphRAG

**GraphRAG** replaces the traditional dense-passage retriever with a **structured knowledge graph (KG)**. Instead of retrieving flat text passages, GraphRAG extracts a **subgraph** of entities and relations relevant to the query answer, then linearizes that subgraph into a textual prompt for a generative model (BART-large).

#### Pipeline

1. **KG Construction**
   - Source: Wikipedia passages (`wiki_dpr` / `psgs_w100`)
   - Relation Extraction: **REBEL** (`Babelscape/rebel-large`) for triple extraction.
   - Dependency Parsing: **SpaCy** (`en_core_web_sm`) for syntactic triples.
   - Output: `subject --[relation]--> object` triples stored in a NetworkX `MultiGraph`.

2. **Subgraph Retrieval**
   - Given an answer entity (e.g., "Albert Einstein"), find the node in the KG.
   - Retrieve its 1-hop or 2-hop neighbors (e.g., `Albert Einstein --[award]--> Nobel Prize`).
   - Convert triples into a natural-language context string.

3. **Generation**
   - **Baseline:** BART-large receives only the entity name.
   - **GraphRAG:** BART-large receives the entity name + the linearized subgraph.

> **Key Difference from standard RAG:** Standard RAG retrieves *unstructured text chunks*. GraphRAG retrieves *structured relational triples*, which often contain more precise and compact factual information.

---

### Coreference Resolution in Knowledge Graphs

Coreference resolution is the NLP task of identifying when multiple expressions in a text refer to the same real-world entity. For example:

> "**Albert Einstein** was born in Germany. **He** later moved to Switzerland. **There** **he** developed the theory of relativity."

Without coreference resolution, the graph would create **four separate nodes**:
- `albert einstein`
- `he`
- `there`
- `he` (again)

With coreference resolution (using **FastCoref / FCoref**), all pronouns are replaced by their antecedent, so every mention points back to `albert einstein`. This produces a **more connected, less fragmented graph**.

#### Comparison: No Coref vs. With Coref

| Metric | **No Coreference** | **With Coreference** |
|---|---|---|
| **Nodes** | 110,921 | 122,577 |
| **Edges** | 96,236 | 107,703 |
| **Connected Components** | 26,605 | (fewer, more merged) |
| **Largest Component** | 47,488 nodes | (larger) |
| **Processing Time** | ~11 hours | ~12+ hours |
| **Entity Aliases** | Separate nodes (`einstein` ≠ `albert einstein`) | Merged (`einstein` → `albert einstein`) |
| **Pronoun Handling** | Pronouns (`he`, `she`, `it`) remain as nodes or are resolved only with simple heuristic memory | Pronouns are replaced by their representative mention via FastCoref |

> **Why it matters:** A graph with coreference resolution has higher semantic coherence. Merging aliases reduces redundancy and strengthens the connectivity of important entities, which improves the quality of retrieved subgraphs during generation.

---

### Graph Comparison: Notebook Outputs & Neo4j Queries

The following subsections reproduce the raw console outputs from the graph-analysis cells in both `kg_without_coref.ipynb` (Section 10) and `kg_with_coref.ipynb` (Section 11). For every comparison, an equivalent **Cypher query** is provided so you can import the `.graphml` files into Neo4j and verify the claims yourself.

---

#### 1. Graph Statistics Overview

**No Coref Output (`kg_without_coref.ipynb`)**
```
=== Knowledge Graph WITHOUT Coreference Resolution ===
Total Nodes: 110921
Total Edges (triples): 96236
Average Degree: 1.74
Connected Components: 26605
Largest Component Size: 47488 nodes
```

**With Coref Output (`kg_with_coref.ipynb`)**
```
=== Knowledge Graph with Coreference Resolution ===
Total Nodes: 122577
Total Edges (triples): 107703
Average Degree: 1.76
Connected Components: 28398
Largest Component Size: 54191 nodes
```

**Analysis**
Coref increased total nodes and edges by ~10%, but the largest connected component grew by **14%** (47,488 → 54,191). Average degree barely changed (1.74 → 1.76). Counter-intuitively, connected components also rose slightly (26,605 → 28,398); coref merges pronouns into the giant component, yet it also introduces more unique resolved names that remain isolated in smaller article-specific clusters.

**Equivalent Neo4j Cypher Query**
```cypher
// Overall graph statistics
MATCH (n)
OPTIONAL MATCH (n)--()
WITH count(DISTINCT n) AS totalNodes, count(*) AS totalRelationships
RETURN
  totalNodes,
  totalRelationships,
  round(100.0 * totalRelationships / totalNodes, 2) AS densityRatio
```
*For connected-component sizes, install the Neo4j GDS library and run `gds.wcc.stream('myGraph')` (Weakly Connected Components).*

---

#### 2. The "Which" Problem — Top 10 Most Connected Nodes

**No Coref Output**
```
Top 10 Most Connected Nodes:
  which: 0.0161
  that: 0.0091
  the united states: 0.0068
  who dey: 0.0050
  world war ii: 0.0028
  the world health organization who: 0.0024
  what: 0.0022
  new york: 0.0019
  new york city: 0.0014
  the soviet union: 0.0014
```

**With Coref Output**
```
Top 10 Most Connected Nodes:
  which: 0.0117
  that: 0.0066
  the united states: 0.0062
  the world health organization who: 0.0051
  who dey: 0.0030
  world war ii: 0.0026
  new york: 0.0017
  what: 0.0016
  new york city: 0.0014
  the soviet union: 0.0013
```

**Analysis**
Even after coref, `which` and `that` remain the top hubs—FastCoref does **not** fully eliminate determiners. However, their centrality dropped significantly (`which`: 0.0161 → 0.0117, a **27%** reduction; `that`: 0.0091 → 0.0066, a **27%** reduction). Real-world entities (`the united states`, `world war ii`, `new york`) maintain similar rankings, proving the graph backbone is robust.

**Equivalent Neo4j Cypher Query**
```cypher
// Top 10 nodes by degree centrality
MATCH (n)
OPTIONAL MATCH (n)--()
RETURN n.label AS entity, count(*) AS degree
ORDER BY degree DESC
LIMIT 10
```

---

#### 3. Triple Source Distribution

**No Coref Output**
```
Triple Sources:
source
dependency    71350
rebel         28161
```

**With Coref Output**
```
Triple Sources:
source
dependency    80164
rebel         33436
```

**Analysis**
Coref increased both dependency-parsed and REBEL-extracted triples by ~12%. Resolving pronouns to named entities means fewer triples are discarded by the pronoun blacklist (`BAD_PRONOUNS`), so more triples survive validation and enrich the graph.

**Equivalent Neo4j Cypher Query**
```cypher
// Count triples by extraction source
MATCH ()-[r]->()
RETURN r.sources AS source, count(*) AS tripleCount
ORDER BY tripleCount DESC
```

---

#### 4. Sample Triples — Entity Fragmentation vs. Resolution

**No Coref Sample (`kg_without_coref.ipynb`, Section 11)**
```
1.  [dependency] lynne spears --[negotiate]--> her manager
7.  [dependency] lynne spears --[travel]--> sweden
8.  [dependency] lynne spears --[ask]--> larry rudolph
11. [dependency] lynne spears --[travel]--> new york
14. [dependency] lynne spears --[fly]--> cheiron studios
15. [dependency] mikey bassie --[hear]--> spears
```

**With Coref Sample (`kg_with_coref.ipynb`, Section 12)**
```
1.  [rebel]    britney spears --[mother]--> lynne spears
2.  [rebel]    lynne spears --[child]--> britney spears
3.  [dependency] britney spears --[negotiate]--> american pop singer britney spears manager
8.  [dependency] britney spears --[travel]--> sweden
12. [dependency] britney spears --[travel]--> new york
15. [dependency] britney spears --[fly]--> cheiron studios
```

**Analysis**
This is the most dramatic semantic difference. In the no-coref graph, pronouns and fragmented aliases cause **misattribution**: actions that belong to Britney Spears are attached to `lynne spears` or the orphaned fragment `spears`. After coref, FastCoref resolves "she" → `britney spears`, so the subject of `travel`, `fly`, and `negotiate` shifts from the mother to the daughter. The with-coref graph also discovers **new** REBEL relations (`mother`, `child`) that were invisible before because the entities were split.

**Equivalent Neo4j Cypher Query**
```cypher
// Find all relations for a specific entity family to see attribution shifts
MATCH (n)-[r]->(m)
WHERE n.label CONTAINS 'britney spears'
   OR n.label CONTAINS 'lynne spears'
   OR n.label CONTAINS 'spears'
   OR m.label CONTAINS 'britney spears'
   OR m.label CONTAINS 'lynne spears'
   OR m.label CONTAINS 'spears'
RETURN
  n.label AS subject,
  r.relation AS predicate,
  m.label AS object,
  r.sources AS source,
  r.confidence AS confidence
ORDER BY subject, predicate
```

---

#### 5. The "Spears" Fragmentation Problem

**No Coref:** The node `spears` appears as a disconnected last-name fragment (`mikey bassie --[hear]--> spears`). It is **not** linked to `britney spears` or `lynne spears`.

**With Coref:** The fragment disappears because partial mentions are resolved to the full canonical name `britney spears`.

**Analysis**
This demonstrates the alias-merge benefit of coref. In a downstream GraphRAG retrieval task, a query for `britney spears` would **miss** the no-coref edge `mikey bassie --[hear]--> spears`, yielding an incomplete subgraph and a weaker prompt for the generator.

**Equivalent Neo4j Cypher Query**
```cypher
// Find isolated name fragments and pronoun garbage
MATCH (n)
WHERE n.label = 'spears'
   OR n.label = 'he'
   OR n.label = 'she'
   OR n.label = 'her'
   OR n.label = 'him'
OPTIONAL MATCH (n)--()
RETURN n.label AS fragment, count(*) AS degree
ORDER BY degree DESC
```

---

## Datasets

| Dataset | Purpose | Size |
|---|---|---|
| **SearchQA** (`search_qa`) | Fine-tuning RAG-Token & RAG-Sequence | 151k train / 43k test / 21k val |
| **Wiki-DPR** (`wiki_dpr/psgs_w100`) | Building the knowledge graph | 133,856 passages (3 splits) |

For RAG fine-tuning, the input is the **answer** and the target is the **question** (Jeopardy clue).

---

## Models & Pipelines

### 1. RAG-Token
- **Notebook:** `rag token/rag_token.ipynb`
- **Model:** `facebook/rag-token-nq`
- **Retrieved Docs:** 10 (`n_docs=10`)
- **Training:** 2 epochs, batch size 2, gradient accumulation 8, LR `1e-5`
- **Evaluation:** Q-BLEU-1 = **19.10**, BLEU-1 = **14.20**

### 2. RAG-Sequence
- **Notebook:** `rag sequence/rag_seq.ipynb`
- **Model:** `facebook/rag-sequence-nq`
- **Retrieved Docs:** 10 (`n_docs=10`)
- **Training:** 2 epochs, same hyperparameters
- **Evaluation:** Q-BLEU-1 = **18.10**, BLEU-1 = **10.00**

### 3. GraphRAG (No Coref)
- **Builder:** `knowledge graph/kg_without_coref.ipynb`
- **Evaluator:** `knowledge graph/bart_eval_no_coref.ipynb`
- **Generator:** `facebook/bart-large-cnn` (fine-tuned)
- **Graph:** 110,921 nodes, 96,236 edges
- **Relation Sources:** 71,350 dependency-parsed triples + 28,161 REBEL triples

### 4. GraphRAG (With Coref)
- **Builder:** `knowledge graph/kg_with_coref.ipynb`
- **Evaluator:** `knowledge graph/bart_eval_with_coref.ipynb`
- **Generator:** `facebook/bart-large-cnn` (fine-tuned)
- **Graph:** 122,577 nodes, 107,703 edges
- **Coreference Tool:** `fastcoref` (FCoref model)

---

## Evaluation Metrics

We use the **Answerability Metric (Q-BLEU)** from Nema & Khapra (EMNLP 2018), implemented in `Answerability-Metric/`.

Q-BLEU combines:
- **N-gram overlap** (BLEU)
- **Named Entity overlap** (NER)
- **Question type matching** (QT)
- **Relevance scoring** (RE)

Example evaluation command:
```bash
python Answerability-Metric/answerability_score.py \
  --data_type squad \
  --ref_file references.txt \
  --hyp_file hypotheses.txt \
  --ner_weight 0.41 \
  --qt_weight 0.20 \
  --re_weight 0.36 \
  --delta 0.66 \
  --ngram_metric Bleu_1
```

---

## Results Summary

| Model | Q-BLEU-1 | BLEU-1 |
|---|---|---|
| **RAG-Token** | 19.10 | 14.20 |
| **RAG-Sequence** | 18.10 | 10.00 |
| **GraphRAG (No Coref)** | (see eval notebook) | (see eval notebook) |
| **GraphRAG (With Coref)** | (see eval notebook) | (see eval notebook) |

> Note: The RAG models were trained on SearchQA and evaluated on the validation split. GraphRAG results are produced by BART-large prompted with KG subgraphs; exact scores depend on subgraph retrieval depth and prompt formatting.

---

## How to Run

### RAG-Token or RAG-Sequence
1. Open `rag token/rag_token.ipynb` or `rag sequence/rag_seq.ipynb`.
2. Install dependencies (the notebook pins `transformers==4.30.0`, `torch==2.2.2`, etc.).
3. Run all cells to fine-tune and evaluate.

### Knowledge Graph Construction
1. Run `knowledge graph/preparation/prepare_data_splits.ipynb` to split `wiki_dpr` into 3 parts.
2. Open `kg_without_coref.ipynb` or `kg_with_coref.ipynb`.
3. Run all cells; the pipeline will process each part and save:
   - `knowledge_graph_no_coref.graphml` / `.pkl`
   - `knowledge_graph_with_coref.graphml` / `.pkl`

### GraphRAG Evaluation
1. Open `bart_eval_no_coref.ipynb` or `bart_eval_with_coref.ipynb`.
2. Load the corresponding `.graphml` file.
3. Run baseline and GraphRAG generation cells.
4. Export `references.txt` and `hypotheses.txt` for Q-BLEU scoring.

---

## References

- **RAG:** Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- **Q-BLEU / Answerability:** Nema, P., & Khapra, M. (2018). *Towards a Better Metric for Evaluating Question Generation Systems*. EMNLP.
- **REBEL:** Huguet Cabot, P., & Navigli, R. (2021). *REBEL: Relation Extraction By End-to-end Language generation*. EMNLP.
- **FastCoref:** Otmazgin, S., et al. (2022). *FastCoref: Fast and Accurate Coreference Resolution*.
- **DPR / Wiki-DPR:** Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP.
- **SearchQA:** Dunn, M., et al. (2017). *SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine*.

---

## License

This project is for academic and research purposes. Please refer to the individual model licenses (Hugging Face Transformers, RAG checkpoints, BART, REBEL, FastCoref) for commercial use restrictions.
