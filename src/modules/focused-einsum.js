// Module: Einsum Notation
// Section 0.1: Einstein summation from first principles through applied patterns
// Single-concept module: einsum fluency for deep learning
// Proper learning module with alternating info/mc steps

export const einsumLearning = {
  id: "0.1-einsum-learning-easy",
  sectionId: "0.1",
  title: "Einsum Notation",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "What is Einsum?",
      content: "**Einsum** (Einstein summation) is a compact notation for expressing operations on tensors — vectors, matrices, and higher-dimensional arrays. It originates from physics, where Einstein proposed the convention that repeated indices are implicitly summed over.\n\nIn deep learning frameworks like PyTorch and NumPy, einsum is written as a string that describes the operation:\n\n```\ntorch.einsum('index_pattern', tensor_A, tensor_B)\n```\n\nThe core rule is simple:\n- Each tensor gets a set of **index labels** (one per dimension)\n- Indices that appear in the **inputs but not the output** are **summed over**\n- Indices that appear in the **output** are **kept**\n\nFor example, `'ij,jk->ik'` means: tensor A has indices $i, j$; tensor B has indices $j, k$; the output has indices $i, k$. Since $j$ appears in the inputs but not the output, it is summed over. This gives $C_{ik} = \\sum_j A_{ij} B_{jk}$ — matrix multiplication.\n\nEinsum can express almost any tensor operation in a single line, making it invaluable for reading and writing deep learning code."
    },
    {
      type: "mc",
      question: "In einsum notation `'ij,jk->ik'`, which index is summed over and why?",
      options: ["Index $i$, because it appears first in the pattern and first indices are always contracted", "Index $k$, because it only appears in the second tensor and single-tensor indices are summed", "Index $j$, because it appears in both inputs but not in the output", "All indices are summed, producing a scalar output from the contraction"],
      correct: 2,
      explanation: "The einsum rule: indices present in the inputs but absent from the output are summed over. Here $j$ appears in both `ij` and `jk` but not in the output `ik`, so it is summed: $\\sum_j A_{ij} B_{jk}$. Indices $i$ and $k$ appear in the output, so they are kept as free dimensions of the result."
    },
    {
      type: "info",
      title: "Building Blocks: Common 2D Patterns",
      content: "Here are the fundamental einsum patterns for matrices, building from simplest to most complex:\n\n**Matrix multiply**: `'ij,jk->ik'`\n$$C_{ik} = \\sum_j A_{ij} B_{jk}$$\nThe shared index $j$ is contracted (summed over).\n\n**Element-wise (Hadamard) product**: `'ij,ij->ij'`\n$$C_{ij} = A_{ij} \\cdot B_{ij}$$\nAll indices are kept — no summation, just pointwise multiplication.\n\n**Dot product of vectors**: `'i,i->'`\n$$s = \\sum_i a_i b_i$$\nBoth indices are contracted, producing a scalar (empty output).\n\n**Outer product of vectors**: `'i,j->ij'`\n$$C_{ij} = a_i \\cdot b_j$$\nNo shared indices, so nothing is summed — the result is a matrix.\n\n**Trace of a matrix**: `'ii->'`\n$$s = \\sum_i A_{ii}$$\nThe repeated index in a single tensor means: take the diagonal, then sum.\n\n**Matrix transpose**: `'ij->ji'`\n$$B_{ji} = A_{ij}$$\nJust relabeling the output indices swaps the axes."
    },
    {
      type: "mc",
      question: "What does `torch.einsum('ij,jk->ik', A, B)` compute?",
      options: ["Element-wise (Hadamard) product of $A$ and $B$", "Trace of the matrix product $AB$, producing a scalar", "Outer product of $A$ and $B$, producing a 4D tensor", "Standard matrix product $AB$, contracting over $j$"],
      correct: 3,
      explanation: "The index $j$ appears on both inputs but not in the output, so it is summed over: $\\sum_j A_{ij} B_{jk} = (AB)_{ik}$. This is standard matrix multiplication. Einsum makes the contraction axis explicit — `ij,jk->ik` is the canonical matrix multiply pattern. The element-wise product would be `'ij,ij->ij'` (same indices, all kept)."
    },
    {
      type: "mc",
      question: "What does `torch.einsum('ij,ij->', A, B)` compute, where $A$ and $B$ are matrices?",
      options: ["The matrix product $AB$, a matrix", "The element-wise product $A \\odot B$, a matrix", "The trace of $A$, a scalar", "The sum of all elements of the element-wise product: $\\sum_{i,j} A_{ij} B_{ij}$, a scalar"],
      correct: 3,
      explanation: "Both $i$ and $j$ appear in the inputs but the output is empty (scalar), so both are summed over: $\\sum_{i,j} A_{ij} B_{ij}$. This is the Frobenius inner product of two matrices — element-wise multiply, then sum everything. It generalizes the vector dot product to matrices. This equals $\\text{tr}(A^\\top B)$."
    },
    {
      type: "info",
      title: "Adding a Batch Dimension",
      content: "Deep learning operates on **batches** — many inputs processed simultaneously. Einsum handles this naturally by adding a batch index that appears in all tensors (so it's never summed over).\n\n**Batched matrix multiply**: `'bij,bjk->bik'`\n$$C_{bik} = \\sum_j A_{bij} B_{bjk}$$\n\nIndex $b$ appears everywhere (inputs and output), so it's kept — the operation is performed independently for each element in the batch. Index $j$ is summed over (shared between inputs, absent from output).\n\nThis is equivalent to a loop:\n```\nfor each b:\n    C[b] = A[b] @ B[b]\n```\nbut expressed as a single operation. This is what PyTorch's `torch.bmm` computes.\n\nThe batch index convention generalizes: any index that appears in all tensors and the output is a \"free\" dimension that the operation iterates over. You can have multiple such indices — e.g., $b$ for batch and $h$ for head in multi-head operations."
    },
    {
      type: "mc",
      question: "Which einsum string computes a **batched matrix multiply** $C_{b} = A_{b} B_{b}$ for a batch of matrices?",
      options: ["`'ij,jk->ik'`", "`'bi,bj->bij'`", "`'bij,bjk->bik'`", "`'bij,bkj->bik'`"],
      correct: 2,
      explanation: "`'bij,bjk->bik'`: index $b$ appears in all three tensors so it is kept (not summed); $j$ is summed over as the shared inner dimension. The result is $C_{bik} = \\sum_j A_{bij} B_{bjk}$ — independently multiplying the $b$-th matrix pair. Option D (`'bij,bkj->bik'`) would compute $\\sum_j A_{bij} B_{bkj}$, which is $A_b B_b^\\top$ (note $B$'s indices are transposed) — a common mistake."
    },
    {
      type: "info",
      title: "Multi-Head Attention in Einsum",
      content: "One of the most important applications of einsum is expressing **multi-head attention** — the core operation in transformer models. Here's how it works.\n\nThe tensors involved have 4 dimensions with indices:\n- $b$ = batch (which input in the batch)\n- $h$ = head (which attention head)\n- $i$ or $j$ = sequence position (which token)\n- $d$ = head dimension (dimension within each head)\n\nAttention has two main steps:\n\n**Step 1 — Compute attention scores**: For each query position $i$, compute a dot product with every key position $j$ across the head dimension $d$:\n\n`'bhid,bhjd->bhij'` → $\\text{scores}_{bhij} = \\sum_d Q_{bhid} K_{bhjd}$\n\nThe $d$ index is summed (dot product). The result is a score for each (query, key) pair.\n\n**Step 2 — Weighted sum of values**: After applying softmax to the scores, weight the value vectors by the attention scores:\n\n`'bhij,bhjd->bhid'` → $\\text{out}_{bhid} = \\sum_j \\text{attn}_{bhij} V_{bhjd}$\n\nThe $j$ index is summed (aggregate over all positions). For each query position $i$, this produces a weighted average of the value vectors."
    },
    {
      type: "mc",
      question: "What does `torch.einsum('bhij,bhjd->bhid', attn_weights, V)` compute? (Indices: $b$=batch, $h$=head, $i$=query position, $j$=key position, $d$=head dim)",
      options: ["The weighted sum of values: for each query position $i$, sum $V$ weighted by attention scores over $j$", "The dot-product attention logits: a score for every (query, key) pair across the head dimension", "The outer product of attention weights and values, producing a 5D tensor", "A normalization operation that scales the values by the sum of attention weights"],
      correct: 0,
      explanation: "Index $j$ is summed over (it appears in both inputs but not the output), giving $\\text{output}_{bhid} = \\sum_j \\text{attn}_{bhij} \\cdot V_{bhjd}$. For each batch $b$, head $h$, and query position $i$, this computes a weighted sum of the value vectors $V_{bhjd}$ across all key/value positions $j$, with weights given by the attention scores. This is the core \"aggregate information\" step of attention."
    },
    {
      type: "info",
      title: "Reading Einsum: A Systematic Approach",
      content: "When you encounter an einsum string in code, here's how to decode it:\n\n1. **List all indices** that appear. For `'bhid,bhjd->bhij'`: indices are $b, h, i, d, j$.\n\n2. **Identify free indices** (in the output): $b, h, i, j$ — these are the dimensions of the result.\n\n3. **Identify contracted indices** (in inputs but not output): $d$ — these are summed over.\n\n4. **Write the formula**: $\\text{result}_{bhij} = \\sum_d \\text{input1}_{bhid} \\cdot \\text{input2}_{bhjd}$\n\n5. **Interpret**: This is a dot product over the $d$ dimension between vectors at position $i$ (from input1) and position $j$ (from input2), done independently for each batch $b$ and head $h$.\n\nCommon patterns to recognize:\n- **Same index in two inputs, not in output** → dot product / contraction along that dimension\n- **Index in output but only one input** → broadcasting (that input is replicated)\n- **Index in all places** → batch/free dimension (no computation, just iteration)"
    },
    {
      type: "mc",
      question: "The full scaled dot-product attention is $\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V$. Using indices $b, h, i, j, d$: which sequence of einsum calls correctly computes the attention output?",
      options: ["`scores = einsum('bhid,bhid->bhi', Q, K)` then `out = einsum('bhi,bhid->bhid', softmax(scores/\\sqrt{d}), V)`", "`scores = einsum('bhid,bhjd->bhij', Q, K)` then `out = einsum('bhij,bhjd->bhid', softmax(scores/\\sqrt{d}), V)`", "`scores = einsum('bhid,bhjd->bhid', Q, K)` then `out = einsum('bhij,bhjd->bij', softmax(scores/\\sqrt{d}), V)`", "`scores = einsum('bhij,bhjd->bhid', Q, K)` then `out = einsum('bhid,bhjd->bhij', softmax(scores/\\sqrt{d}), V)`"],
      correct: 1,
      explanation: "Step 1: `'bhid,bhjd->bhij'` — $d$ is summed (dot product over head dim between each query position $i$ and key position $j$), giving attention logits of shape $(b, h, i, j)$. Step 2: after softmax, `'bhij,bhjd->bhid'` — $j$ is summed (weighted sum of values over key positions), giving output of shape $(b, h, i, d)$. First contract over $d$ to get scores, then contract over $j$ to aggregate values."
    },
    {
      type: "info",
      title: "Why Einsum Matters",
      content: "You might wonder: why not just use `torch.matmul` or `@`? For 2D matrix multiplication, those work fine. But einsum shines for operations that are awkward to express with standard linear algebra:\n\n- **Batched operations with multiple batch dims** (batch + head): `'bhid,bhjd->bhij'`\n- **Contracting over non-standard axes**: no need to transpose/reshape first\n- **Complex tensor operations in one line**: what would take `permute → reshape → matmul → reshape → permute` becomes a single einsum string\n\nPractically, einsum is the standard way to express attention, tensor products, and multi-head operations in research code. When reading ML papers with code, you'll encounter einsum constantly. The ability to read an einsum string and immediately understand what it computes — what's contracted, what's batched, what the output shape is — is a core skill for working with modern deep learning code."
    }
  ]
};
