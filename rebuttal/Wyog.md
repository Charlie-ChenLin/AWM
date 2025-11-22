# Reply

We thank the reviewer for the positive assessment of AWM, including its strong empirical performance, efficiency, and the comprehensive analysis of manipulations. In what follows, we respond to each concern raised by the reviewer.

> **[W1&Q1]** AWM requires direct access to model weights, limiting its applicability. Can AWM be extended to handle other forms of misuse such as model distillation?

We are grateful to the reviewer for this clarifying question. As discussed in Section 4, AWM is explicitly designed for the white-box setting like REEF and HuRef, where one suspects that a model directly or indirectly inherits weights from a base model, and in this regime weight-matrix similarity is the right signal to examine. By contrast, pure API-based knowledge distillation can produce students that mimic the teacher’s behaviour while having almost uncorrelated weights, so we view these cases as primarily the domain of black-box fingerprinting methods. That said, the two approaches are complementary: lightweight black-box probes can be used to broadly flag potential distillation or copying cases, and, whenever model weights later become available (e.g., for an open model or in an audit), AWM can serve as a high-precision white-box check on weight inheritance. In particular, AWM’s strict control of false positives makes it well suited for third-party verification settings, where regulators, courts, or other independent auditors confidentially examine model weights to assess potential infringement.

> **[W2]** Manipulations in Section 4 are not tested against AWM.

We agree that explicitly evaluating the discussed manipulations is important for closing the gap in our argument. In addition to the analysis in Section 4, we now empirically test the four manipulation strategies considered in Section 4.5, including permutations, signature matrix multiplications, constant scaling, and orthogonal transformations, and report the AWM-detcetd similarity in what follows:

|Attack/Model|Llama-2-7B|Qwen2-7B|Mistral-7B|Llama-3-8B|Gemma-7B|
|-|-|-|-|-|-|
|Permutation|100%|100%|100%|100%|100%|
|Signature|100%|100%|100%|100%|100%|
|Constant Scaling|100%|100%|100%|100%|100%|
|Orthogonal Transformations|100%|100%|100%|100%|100%|

As can be seen above, AWM successfully detects the manipulations discussed in Section 4. These results support the arguments of the paper.