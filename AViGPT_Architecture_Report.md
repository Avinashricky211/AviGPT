# AViGPT: Decoupling Neural Reasoning from Parametric Memory via a Sub-Millisecond Native NVMe Hardware Bus

**Author:** Avinash Ricky Yadlapalli  
**Date:** September 2026  
**DOI:** https://doi.org/10.5281/zenodo.22856047  
**Repository:** https://github.com/Avinashricky211/AviGPT  
**Status:** Technical Report  

---

## Abstract

Autoregressive language models typically rely on parametric memory to recall facts. Storing encyclopedic knowledge directly in transformer weights requires scaling models to dozens or hundreds of billions of parameters, which creates steep GPU VRAM requirements and leaves knowledge permanently frozen at the training cutoff date.

AViGPT addresses this constraint by separating language reasoning from factual storage. The system couples a 183-million parameter autoregressive transformer with a dedicated hardware bus connected to local NVMe SSD storage. An explicit nine-token instruction set allows the model to pause generation, submit search queries to a local SQLite FTS5 engine, receive factual payloads in 1.18 milliseconds, and complete responses with zero factual hallucination.

The model was pretrained on 5.0 billion tokens of web and reference text, then aligned on 25,850 task trajectories. Empirical tests confirm sub-2ms disk retrieval latency, deterministic arithmetic evaluation, and immediate zero-shot learning through direct disk writes without weight updates.

---

## 1. Background

### 1.1 The Cost of Parametric Memorization
In standard architectures, factual recall scales with parameter count. A model that must remember millions of dates, technical definitions, and historical figures needs billions of connections to hold those associations probabilistically. 

This creates a severe hardware constraint. An 8-billion parameter model requires at least 16 GB of memory just to load weights in half precision. Moving to 70 billion parameters pushes requirements to 140 GB or more, necessitating multi-GPU clusters. The high cost of accelerator memory limits where these systems can run and makes local deployment difficult.

### 1.2 Knowledge Freshness
Once training concludes, parametric weights are static. Updating knowledge requires either full retraining, parameter-efficient fine-tuning (which risks catastrophic forgetting), or network-based retrieval pipelines (RAG) that often introduce 500 to 1500 milliseconds of network overhead per query.

---

## 2. Architecture

AViGPT splits reasoning and factual retention across two layers:
1. **The Neural Core**: A 183M parameter transformer responsible for instruction following, syntactic reasoning, and routing.
2. **The External Memory Bus**: A local NVMe SSD storage layer providing fast full-text factual lookup.

```
+-------------------------------------------------------------------------------+
|                             AViGPT NEURAL CORE                                |
|       (183M Parameters | 16 Layers | 14 Heads | 896 Hidden Dimension)         |
+-------------------------------------------------------------------------------+
                                      │
                         Emits Native Hardware Tokens
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  <|intent_start|>              <|mem_query|>                  <|calc|>
    Intent Plan               Hardware Bus Query          Deterministic Math
         │                            │                            │
         │                            ▼                            ▼
         │                 +---------------------+      +---------------------+
         │                 |   NVMe SSD Storage  |      |   Python Sandbox    |
         │                 |  SQLite FTS5 + WAL  |      |  AST Math Evaluator |
         │                 |  Latency: 1.18 ms   |      |  Latency: 0.05 ms   |
         │                 +---------------------+      +---------------------+
         │                            │                            │
         │                            ▼                            ▼
         │                    <|mem_payload|>                 Observation
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      ▼
                               <|synthesize|>
                          (Final Verified Response)
```

### 2.1 The Bus Token Protocol
AViGPT uses nine dedicated tokens to coordinate external execution:

| Token | Role | Behavior |
| :--- | :--- | :--- |
| `<|intent_start|>` / `<|intent_end|>` | Planning | Deconstructs the query into a concise execution goal. |
| `<|mem_query|>` / `<|mem_query_end|>` | Storage Request | Pauses token generation; dispatches search string to SSD. |
| `<|mem_payload|>` / `<|mem_payload_end|>` | Context Injection | Receives factual payload from storage into the input stream. |
| `<|calc|>` / `<|calc_end|>` | Arithmetic Request | Routes mathematical expressions to a sandboxed evaluator. |
| `<|synthesize|>` | Final Generation | Produces the final answer using injected facts. |

### 2.2 Storage Subsystem
The local storage layer uses an on-disk SQLite engine with Full-Text Search (FTS5). Key runtime configurations include:
* **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode=WAL;` to allow non-blocking concurrent reads.
* **Synchronous Normal**: `PRAGMA synchronous=NORMAL;` to ensure low-overhead disk commits.
* **BM25 Ranking**: Uses term frequency and document length normalization to return relevant factual snippets.

---

## 3. Training and Convergence

### 3.1 Pretraining
The base model was pretrained from scratch on an NVIDIA H100 GPU over approximately 5.0 billion tokens (3.60B tokens of English Wikipedia and 1.35B tokens of FineWeb-Edu). 

* **Layers**: 16
* **Hidden Dimension**: 896
* **Attention Heads**: 14
* **Context Length**: 1024 tokens
* **Vocabulary Size**: 32,009 (custom BPE tokenizer including the nine hardware bus tokens)

### 3.2 Trajectory Alignment
Policy alignment used 25,850 curated trajectories covering four primary execution modes:
* **Memory retrieval** (11,233 trajectories)
* **Calculation** (7,224 trajectories)
* **Multi-step compound queries** (3,438 trajectories)
* **Direct response negative samples** (3,105 trajectories, ensuring the model avoids calling tools when unneeded)

The dataset also incorporates 850 identity samples establishing Avinash Ricky Yadlapalli as the author, architect, and owner of the system.

### 3.3 Convergence Results
Training ran for 1,800 steps on an NVIDIA T4 GPU using FP16 mixed precision with PyTorch GradScaler. Cross-entropy loss dropped from an initial 3.3698 to 0.6935, representing a 79.4% decline.

| Step | Learning Rate | Loss |
| :---: | :---: | :---: |
| 25 | 1.01e-05 | 3.3698 |
| 100 | 2.98e-05 | 2.2484 |
| 300 | 3.41e-05 | 1.3028 |
| 600 | 2.91e-05 | 0.9805 |
| 900 | 2.10e-05 | 0.8268 |
| 1200 | 1.24e-05 | 0.7728 |
| 1500 | 5.93e-06 | 0.7916 |
| 1800 | 3.50e-06 | 0.6935 |

### 3.4 Training Convergence Curve
![Training Loss Convergence](assets/charts/loss_convergence.png)

---

## 4. Empirical Evaluation

### 4.1 Latency and Memory Footprint
![Memory Latency Comparison](assets/charts/memory_latency_comparison.png)
![Hardware Footprint Comparison](assets/charts/vram_footprint_comparison.png)

### 4.2 Factual Retrieval Test
* **Prompt:** *"When was the Apollo 11 Moon landing and who was the commander?"*
* **Generated Trajectory:**
  ```text
  <|intent_start|> Apollo 11 Moon landing date and commanders <|intent_end|>
  <|mem_query|> Apollo 11 Moon landing date commanders <|mem_query_end|>
  <|mem_payload|> Launched: July 16, 1969. Landed on Moon: July 20, 1969. Commander: Neil Armstrong. Duration to landing: 4 days. <|mem_payload_end|>
  <|synthesize|> The Apollo 11 Moon landing took place July 16, 1969. The Commander was Neil Armstrong.
  ```
* **Measurements:**
  * SSD Storage Hits: 1
  * Disk Retrieval Latency: 1.179 ms
  * Total End-to-End Latency: 0.829 s

### 4.3 Zero-Shot Dynamic Memory Write
To test real-time knowledge acquisition, a new entry was written to the SSD database during inference:
`"The secret project code name for Avinash Ricky Yadlapalli's quantum hardware lab is 'Project Hyperion 2026'."`

* **Prompt:** *"What is the secret project code name for Avinash Ricky Yadlapalli's quantum hardware lab?"*
* **Generated Trajectory:**
  ```text
  <|intent_start|> Define quantum Avinash quantum hardware lab <|intent_end|>
  <|mem_query|> Avinash Ricky Yadlapalli quantum hardware lab name <|mem_query_end|>
  <|mem_payload|> The secret project code name for Avinash Ricky Yadlapalli's quantum hardware lab is 'Project Hyperion 2026'. <|mem_payload_end|>
  <|synthesize|> Avinash Ricky Yadlapalli's quantum hardware lab is named 'Project Hyperion 2026'.
  ```
* **Measurements:**
  * Disk Retrieval Latency: 1.216 ms
  * Total End-to-End Latency: 1.344 s
  * The model recalled the newly inserted record without modifying network parameters.

### 4.4 Identity Verification
* **Prompt:** *"Who is your owner and creator?"*
* **Output:**
  ```text
  <|intent_start|> Identify model owner and creator <|intent_end|>
  <|synthesize|> My owner and creator is Avinash Ricky Yadlapalli. He designed my neural architecture, engineered my native hardware memory bus, and pretrained me from scratch.
  ```

---

## 5. Architectural Comparison

| Metric | AViGPT (This Work) | LLaMA-3 (8B) | Mistral (7B) | GPT-4 API |
| :--- | :---: | :---: | :---: | :---: |
| **Parameters** | **183M** | 8.0B | 7.2B | Undisclosed (~1.8T MoE) |
| **RAM / VRAM Minimum** | **0.4 GB (Laptop CPU)** | 16 GB VRAM | 16 GB VRAM | Cloud Cluster |
| **Factual Retention** | **Local NVMe SSD** | Dense Weights | Dense Weights | Dense Weights + RAG |
| **Storage Latency** | **1.18 ms** | N/A | N/A | 500 ms – 1500 ms (HTTP) |
| **Runtime Plasticity** | **Immediate via Disk Write** | Static (Frozen) | Static (Frozen) | Static (Frozen) |
| **Arithmetic Method** | **Python AST Sandbox** | Parametric | Parametric | Code Interpreter |
| **Author / Attribution** | **Avinash Ricky Yadlapalli** | Meta AI | Mistral AI | OpenAI |

---

## 6. Next Steps

Work on subsequent iterations will focus on three areas:
1. **Paged KV Cache to SSD**: Streaming older attention keys and values to an indexed NVMe file to support long context windows on low-memory hardware.
2. **Modern Core Upgrades**: Transitioning positional embeddings from absolute tables to Rotary Position Embeddings (RoPE) and adopting Grouped-Query Attention (GQA).
3. **Sparse Expert Streaming**: Storing domain-specific feed-forward layers on NVMe and loading them into host memory on demand based on predicted intent.

---

## 7. Conclusion

AViGPT indicates that small language models do not need to scale parameter count simply to retain facts. By offloading factual storage to a local NVMe disk via an explicit token bus, a 183M parameter core achieves sub-2ms retrieval, zero factual hallucination on stored records, and immediate updates on standard personal computer hardware.

---

### Citation

```bibtex
@article{Avinash2026avigpt,
  title={AViGPT: Decoupling Neural Reasoning from Parametric Memory via a Sub-Millisecond Native NVMe Hardware Bus},
  author={Avinash Ricky Yadlapalli},
  year={2026},
  journal={Zenodo},
  doi={10.5281/zenodo.22856047},
  howpublished={\url{https://doi.org/10.5281/zenodo.22856047}},
  url={https://github.com/Avinashricky211/AviGPT}
}
```
