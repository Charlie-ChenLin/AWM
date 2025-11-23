# Reply (1/2)
We thank the reviewer for the careful reading of our work and for highlighting its strengths, including novelty, performance and theoretical analysis. In what follows, we will respond to your concerns and revise the paper accordingly.

> **[W1]** Dependency on Word Embeddings: retrianing or replacing the embedding layer with other layers freezed could potentially defeat AWM.

We appreciate the reviewer’s thoughtful comment regarding this issue. 
- First, **retraining the embedding layer from scratch may degrade the model's performance significantly**, which may contradict with the purpose to reuse the model's performance. We conduct an experiment on Llama3-3B, where the embedding layer is re-initialized and trained for 6B tokens in the SlimPajama dataset with other layers freezed. We observe a significant drop in the model's performance on popular benchmarks, arguing the validity of such a manipulation approach. The performance comparison is presented in the following table:

| Model\Benchmark | LAMBADA (OpenAI version) | ARC-E | LAMBADA (Standard version) | ARC-C | WinoGrande | PIQA | HellaSwag | SciQ  | RACE | Average |
|-|-|-|-|-|-|-|-|-|-|-|
|Re-initialized embedding, trained on 6B tokens with other layers freezed|3.4|29.9|2.8|20.2 |53.9|55.6|26.2|47.5|23.7|29.2|
|LLaMA3-3B|70.1|74.5|63.7|42.2|69.0|76.8|55.4|95.5|39.4|65.2|

- Second, **replacement of embeddings and the tokenizer can be detected by AWM**. We study a real-world case where the tokenizer may be replaced and retrained. Noticing the heated debate on the similarity between some certain MoE model and Qwen-2.5-14B, we apply AWM to this case. Even though the tokenizers of the two models differ substantially (roughly 78% of their vocabulary tokens are not shared), AWM reports a Z-score of 248.48, flagging the similarity. In other words, heavy replacement of tokenizers and embeddings does not remove the deeper structural alignment that AWM captures in the remaining layers in this case, and the method continues to flag this pair as highly similar. 

- Finally, **AWM does not rely heavily on the overlap of vocabulary tokens and related word embeddings**. To test how much AWM depends on overlapping tokens, we run an ablation over the number of shared tokens and report the results in Appendix F.1. Within such cases, AWM assigns high Z-scores and detects similarity (average absolute Z-score over 100) in most cases even if the number of overlapping tokens is low (~100 tokens), indicating that it does not heavily rely on a large set of aligned word embeddings. We also present the results in the following: 

|Modification\Overlapping Tokens|10|100|500|1000|3000|5000|10000|
|-|-|-|-|-|-|-|-|
|SFT|354.32|354.32|354.32|354.32|354.32|354.32|354.32|
|Continual Pretrain|5.40|147.64|192.08|204.63|208.02|208.21|210.89|
|Upcycling|44.22|248.28|267.94|275.18|281.71|286.05|287.85|
|Multi modal|205.50|332.80|334.52|334.81|334.82|334.82|334.78|
|RL|355.79|355.79|355.79|355.79|355.79|355.79|355.79|
|Pruning|76.46|250.94|255.51|255.87|255.93|256.06|257.72|


# Reply (2/2)
> **[W2]** Limited Detection Scope: other components (e.g., implanting stolen FFN blocks into an MoE model) are unlikely to be detected by AWM.

- First, **AWM‘s effectiveness can extend to other components of a model**, even though it is designed to detect the similarity of Q,K matrices. In the following table we show the average Z-score of FFN matrices, i.e. Gate, Up and Down matrices, where AWM still flags significant similarity.  Even if the offspring model has gone through extensive continual pretraining (e.g., 5.5T tokens for Qwen2.5-Coder-7B), AWM still successfully detects the similarity. 

|Pair|Average Absolute Z-score|
|-|-|
|Llama-2-7B vs Llemma-7B|194.9922|
|Llama-2-7B vs CodeLlama-7B-hf|200.7813|
|Llama-2-7B vs CodeLlama-7B-Python-hf|196.7552|
|Gemma-2B vs CodeGemma-2B|184.7341|
|Gemma-7B vs CodeGemma-7B|256.0771|
|Qwen2.5-7B vs Qwen2.5-Math-7B|190.4809|
|Qwen2-7B vs Qwen2.5-Coder-7B|106.3877|
|Qwen2-7B vs Qwen2-Math-7B|198.5877|
|Llama-2-70B-fp16 vs CodeLlama-70B-hf|191.6779|
|Llama-2-70B-fp16 vs CodeLlama-70B-Python-hf|183.4636|

- Apart from these results, we notice that there's a heated debate on some certain MoE model recently, whose FFN blocks may originate from Qwen-2.5-14B. Hence, we test the similarity between that model's FFN weight matrices and Qwen-2.5-14B's using AWM, and report the similarity scores as follows:

|Component|Dynamic 64 mlp experts, dim=[1344,5120]|||Shared 4 mlp expert dim=[5376,5120]|||
|-|-|-|-|-|-|-|
|**Block**|**Down**|**Up**|**Gate**|**Down**|**Up**|**Gate**|
|Absolute Z-Score|75.20|71.45|121.13|191.38|161.16|222.45|

> **[W3]** Section 4 is confusing.

Thanks for your helpful comments! We've revised the paper accordingly, adding the basic ideas and conclusions at the beginning of Section 4.  Apart from that, we also modify the writing of Section 4 to improve the paper's readability. 

> **[W4]** The Related Work section on fingerprinting is limited, 

Thank you for your helpful feedback! We've enriched the Related Work section, adding more discussions on white-box fingerprint approaches and black-box ones. Please refer to "Related Work" in the revised paper.

