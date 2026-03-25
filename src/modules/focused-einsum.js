// Module: Einsum Notation & Tensor Contractions
// Section 0.1: Basic einsum patterns through multi-head attention
// Single-concept module: einsum fluency for deep learning

export const einsumLearning = {
  id: "0.1-einsum-learning-easy",
  sectionId: "0.1",
  title: "Einsum Notation and Tensor Contractions",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "Einsum: The Universal Tensor Operation",
      content: "**Einsum** (Einstein summation) notation expresses any tensor contraction in a single string. The rule is simple: indices that appear in the inputs but NOT in the output are **summed over**.\n\nExamples:\n- `'ij,jk->ik'` — matrix multiply ($j$ summed)\n- `'ij,ij->i'` — row-wise dot product ($j$ summed)\n- `'ij->'` — sum all elements (both $i$ and $j$ summed)\n- `'bij,bjk->bik'` — batched matrix multiply ($j$ summed, $b$ kept)\n\nEinsum is how modern deep learning frameworks express attention, tensor products, and multi-head operations. Fluency with einsum is essential for reading and writing transformer code."
    },
    {
      type: "mc",
      question: "What does `torch.einsum('ij,jk->ik', A, B)` compute?",
      options: ["Element-wise (Hadamard) product of $A$ and $B$", "Trace of the matrix product $AB$, a scalar", "Outer product of $A$ and $B$, a 4D tensor", "Standard matrix product $AB$, contracting over $j$"],
      correct: 3,
      explanation: "The index $j$ appears on both inputs but not in the output, so it is summed over: $\\sum_j A_{ij} B_{jk} = (AB)_{ik}$. This is exactly matrix multiplication. Einsum notation makes the contraction axis explicit — `ij,jk->ik` is the canonical matrix multiply pattern."
    },
    {
      type: "mc",
      question: "Which einsum string computes a **batched matrix multiply** $C_{b} = A_{b} B_{b}$ for a batch of matrices?",
      options: ["`'ij,jk->ik'`", "`'bi,bj->bij'`", "`'bij,bjk->bik'`", "`'bij,bkj->bik'`"],
      correct: 2,
      explanation: "`'bij,bjk->bik'`: index $b$ appears in all three tensors so it is kept (not summed); $j$ is summed over as the shared inner dimension. The result is $C_{bik} = \\sum_j A_{bij} B_{bjk}$ — independently multiplying the $b$-th matrix pair. This is what PyTorch's `torch.bmm` does under the hood."
    },
    {
      type: "info",
      title: "Einsum for Multi-Head Attention",
      content: "Multi-head attention uses 4D tensors with indices:\n- $b$ = batch, $h$ = head, $i$ = query position, $j$ = key position, $d$ = head dimension\n\nThe two core operations are:\n1. **Attention scores**: $\\text{scores}_{bhij} = \\sum_d Q_{bhid} K_{bhjd}$ → `'bhid,bhjd->bhij'`\n2. **Weighted value sum**: $\\text{out}_{bhid} = \\sum_j \\text{attn}_{bhij} V_{bhjd}$ → `'bhij,bhjd->bhid'`\n\nThe pattern: first contract over $d$ (dot product between query and key), then contract over $j$ (aggregate values weighted by attention)."
    },
    {
      type: "mc",
      question: "What does `torch.einsum('bhij,bhjd->bhid', attn_weights, V)` compute in a multi-head attention layer? (Indices: $b$=batch, $h$=head, $i$=query position, $j$=key position, $d$=head dim)",
      options: ["The weighted sum of values: for each query position $i$, sum $V$ weighted by attention scores over $j$", "The dot-product attention logits $QK^\\top / \\sqrt{d}$ before the softmax normalization step", "The outer product of queries and keys, producing a 4D tensor of pairwise interactions", "Layer normalization applied across the head dimension to stabilize attention outputs"],
      correct: 0,
      explanation: "Index $j$ is summed over (it appears in both inputs but not the output), giving $\\text{output}_{bhid} = \\sum_j \\text{attn}_{bhij} \\cdot V_{bhjd}$. This is exactly the attention output: for each batch $b$, head $h$, and query position $i$, compute a weighted sum of the value vectors $V_{bhjd}$ across all key/value positions $j$, with weights given by the (softmaxed) attention scores."
    },
    {
      type: "mc",
      question: "Scaled dot-product attention computes $\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V$. Using einsum notation with indices $b$ (batch), $h$ (head), $i$ (query pos), $j$ (key pos), $d$ (head dim): which sequence of einsum calls correctly computes the attention output?",
      options: ["`scores = einsum('bhid,bhid->bhi', Q, K)` then `out = einsum('bhi,bhid->bhid', softmax(scores/√d), V)`", "`scores = einsum('bhid,bhjd->bhij', Q, K)` then `out = einsum('bhij,bhjd->bhid', softmax(scores/√d), V)`", "`scores = einsum('bhid,bhjd->bhid', Q, K)` then `out = einsum('bhij,bhjd->bij', softmax(scores/√d), V)`", "`scores = einsum('bhij,bhjd->bhid', Q, K)` then `out = einsum('bhid,bhjd->bhij', softmax(scores/√d), V)`"],
      correct: 1,
      explanation: "Step 1: `'bhid,bhjd->bhij'` — $d$ is summed (dot product over head dim between each query $i$ and key $j$), giving attention logits of shape $(b, h, i, j)$. Step 2: `'bhij,bhjd->bhid'` — $j$ is summed (weighted sum of values over key positions), giving output of shape $(b, h, i, d)$. The full attention pattern is: contract over $d$ to get scores, softmax, then contract over $j$ to aggregate values."
    }
  ]
};
