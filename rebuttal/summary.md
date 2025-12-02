# Summary of Rebuttal for New Area Chair
Dear Area Chair,

Thank you for overseeing the evaluation of our submission. To facilitate your assessment, we first highlight the main strengths identified in the reviews, and then summarize the key concerns together with our responses.
## Key strenghts


1. **Training-free design with zero impact on LLM performance** (@Reviewers tHGT, z54L). Reviewers agree that AWM has no impact on model performance, in contrast to watermarking approaches that require model training.

2. **Effectiveness and robustness** (@Reviewers tHGT, z54L, fYwn, Wyog). Reviewers note that AWM achieves extremely low false‑positive rates (AUC = 1.0, TPR@1% FPR = 100%) across diverse training recipes (SFT, RL, multi‑modal post‑training, continual pretraining) and architectural changes (upcycling, pruning).

3. **Comprehensive theoretical analysis** (@Reviewers fYwn, Wyog). Reviewers recognize that AWM is well motivated by a thorough theoretical analysis of potential attacks.

4. **Computational efficiency** (@Reviewers tHGT, z54L, Wyog). Reviewers praise the lightweight design of AWM, where detection takes at most 30s on a single NVIDIA 3090 GPU per model pair.

## Main concerns and our response

1. **Effective detection in other scenarios**. We show that the effectiveness of AWM extends to MoE models (c.f. response to Reviewer tHGT, W1&Q1), and that AWM remains robust under attacks such as LoRA finetuning and layer addition (c.f. response to Reviewer tHGT, Q2), as well as layer removal, depth/width changes, cross-layer sharing, and block reordering (c.f. Reviewer z54L, W3&Q3). We also demonstrate that AWM can flag cases where a model is fused from mixed sources (c.f. response to Reviewer z54L, W5&Q5), detect FFN similarities in both curated model pairs and a real‑world case (c.f. Reviewer fYwn, W2), and does not heavily rely on shared vocabulary tokens, successfully detecting replacements of embeddings and tokenizers in real‑world cases (c.f. response to Reviewer fYwn, W1).
2. **Robustness against attacks**. We show with concrete examples that AWM remains robust under a variety of attacks, including signature matrices, permutations, constant scaling and orthogonal transformations (c.f. response to Reviewer Wyog, W2). 
3. **More baselines and ablation studies**. We add PCS and Intrinsic Fingerprint in experiments, which are still largely outperformed by AWM. We also conduct ablations on the design of CKA in AWM, covering the selection of kernels and biasedness. (c.f. response to Reviewer z54L, W6&Q7) 

All four reviewers give positive evaluations of our submission, and two reviewers have promised to raise scores before November 27 based on our rebuttal. We sincerely appreciate your time and effort in evaluating our submission, and we are glad to provide any further clarification if needed.

<!-- - clarity in theoretical analysis (@Reviewer fYwn)
- more Related Works (@Reviewer fYwn)
- evaluation sheet (@Reviewer z54L)
- CKA/LAP details (@Reviewer z54L) -->

<!-- @放到每一点里去,(c.f. response to reviewer xxx, W1)
merge 2/3 to 1 -->

<!-- idea novelty (fYwn) -->
<!-- promising results (fYwn) -->
<!-- robust across different settings (z54L) -->

