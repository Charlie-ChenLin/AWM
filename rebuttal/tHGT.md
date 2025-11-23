# Reply
Thank you for your positive assessment of the computational efficiency, training-free design, comprehensive multi-model evaluation, and extensive false-positive analysis in the paper. We appreciate your time and constructive feedback. Below, we provide point-by-point responses to your comments.
> **[W1&Q1]** Will AWM work with larger models and other model architectures, such as MoE?

AWM remains effective in these cases. 
- For larger models, we show in the "Continual Pretrain" part of Table 2 that AWM successfully detects the similarity among Llama2-70B and its continual-pretrained offsprings, CodeLlama-70b-hf and CodeLlama-70b-Python-hf. In both of the two cases,  AWM's absolute Z-scores are over 230, flagging significant similarity. 
- For MoE models, we conduct two lines of tests, dense base models vs their upcycled MoE offsprings,  and MoE base models vs their offsprings. 
  - For the first aspect, we show in the "Upcycling" part of Table 2 that AWM still works. To be specific, we test the similarity of Mistral-7B, Llama3-8B, Llama2-7B, Qwen-1.8B, Minicpm-2B, and their upcycled variants, and find that AWM accurately detects the similarity with absolute Z-scores at least 10 times larger than previous methods.
  - For the second, we compare Qwen3-235B-A22B-FP8 against its offsprings. In such a scenario, AWM still quickly flags the similarity. The results are shown in the following table.

|Base|Qwen3-235B-A22B|||
|-|-|-|-|
|Offspring|Qwen3-235B-A22B-Thinking-2507|Qwen3-VL-235B-A22B-Instruct|Qwen3-235B-A22B-Instruct-2507|
|Absolute Z-score|349.6607|343.9464|350.0179|

> **[Q2]** Any thoughts on how this will work if the post training is done with LoRA methods and or adding new layers of randomly initialized parameters.

AWM remains applicable in both scenarios. 
- When **new layers** are added to the model, **Algorithm 1 first finds an optimal match between layers of the base model and the derived model by performing LAP on layer-wise similarities**. We then compute UCKA on these matched layer pairs, from which AWM successfully recovers the similarity. Empirically, we collect 3 additional model pairs under this regime, and test AWM against them. We also conduct an experiment where  two randomly initialized layers are added to LLaMA-3.2-3B and the new model is continually pretrained with 6B tokens in SlimPajama dataset. We term this model as LLaMA-3.2-3B-two-layers, and use AWM to test its similarity to LLaMA-3.2-3B.  In all of the four cases, AWM successfully identifies the originality of models. We summarize the results in the following table.

Modification|Model Pair|Absolute Z-Score
|-|-|-|
Add layers|Mistral-7B-v0.1 vs SOLAR-10.7B-v1.0|323.5893
| |Yi-6B vs Yi-9B|329.5536
| |Llama2-7b vs LLaMA-Pro-8B|355.8036
| |LLaMA-3.2-3B vs LLaMA-3.2-3B-two-layers|355.1029


- For **LoRA**, which introduces small low-rank updates to weights, we find empirically that AWM is robust to such perturbations. For example, AWM reports 99.81% similarity between the LoRA-fine-tuned Firefly-LLaMA2-13B and its base model LLaMA2-13B (Table 1), and detects the similarity between ChatGLM-6B and its LoRA-fine-tuned variant with a Z-score of 355.27 (Table 2). We additionally collect 5 pairs of models where Lora is applied , and perform AWM on these cases. The results suggest that AWM successfully flags model similarities. We summarize the results as follows: 

Modification|Model Pair|Absolute Z-Score
|-|-|-|
LoRA|Llama2-7b vs Llama-2-Medical-Merged-LoRA|354.9107
| |Llama-7b-hf vs chinese-llama-lora-7b|241.9821
| |Llama-7b-hf vs alpaca-lora-7b|352.8750
| |Llama-13b vs chinese-llama-lora-13b|296.6964
| |Llama-13b vs ziya-llama-13b-medical-lora|354.0179

