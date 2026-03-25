// Focused module: Einsum Notation and Tensor Contractions from first principles.
// Covers the contraction rule, basic patterns, batched operations, and multi-head attention einsums.

export const einsumLearning = {
  id: "0.1-einsum-learning-easy",
  sectionId: "0.1",
  title: "Einsum Notation and Tensor Contractions",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "The Einsum Rule: Summation by Index",
      content: "Einstein summation notation (**einsum**) is a compact way to express tensor operations. The core rule is simple: **any index that appears in the inputs but NOT in the output is summed over**.\n\nAn einsum expression has the form `'input_indices->output_indices'`. For example:\n\n$$\\texttt{'ij,jk->ik'}$$\n\nHere the inputs have indices $i,j$ and $j,k$. The output has indices $i,k$. The index $j$ appears in both inputs but is **absent from the output**, so it gets summed over:\n\n$$C_{ik} = \\sum_j A_{ij} \\, B_{jk}$$\n\nThis is just matrix multiplication. Every einsum expression works the same way: write down which indices label each input tensor, declare which indices survive in the output, and everything else is contracted (summed) away."
    },
    {
      type: "mc",
      question: "Given the einsum expression `'abc,adc->abd'`, which index is summed over (contracted)?",
      options: [
        "Index $a$, because it appears in both inputs and could represent a batch dimension",
        "Index $b$, because it only appears in one of the two input tensors",
        "Index $c$, because it appears in both inputs but is absent from the output",
        "Index $d$, because it appears in only one input and is thus an unmatched free index"
      ],
      correct: 2,
      explanation: "The einsum rule: any index in the inputs but not in the output is summed over. The output indices are $a, b, d$. Index $c$ appears in both inputs ('abc' and 'adc') but is missing from the output 'abd', so $c$ is the contracted index. The result is $Y_{abd} = \\sum_c A_{abc} \\, B_{adc}$."
    },
    {
      type: "info",
      title: "Basic Patterns: Transpose and Trace",
      content: "Einsum can express many familiar operations with no summation at all — or with summation over a single tensor.\n\n**Transpose** — just reorder the output indices:\n$$\\texttt{'ij->ji'} \\quad \\Rightarrow \\quad B_{ji} = A_{ij}$$\n\nNo index is missing from the output, so nothing is summed. We simply rearrange axes.\n\n**Trace** — sum the diagonal of a square matrix:\n$$\\texttt{'ii->'} \\quad \\Rightarrow \\quad \\sum_i A_{ii}$$\n\nThe index $i$ appears twice in the input and is absent from the output (which is a scalar), so it is summed. This produces $A_{00} + A_{11} + \\cdots + A_{nn}$.\n\n**Diagonal extraction** — keep the repeated index:\n$$\\texttt{'ii->i'} \\quad \\Rightarrow \\quad d_i = A_{ii}$$\n\nHere $i$ survives in the output, so no summation occurs — we just extract the diagonal elements."
    },
    {
      type: "mc",
      question: "What does the einsum expression `'ij->j'` compute for a matrix $A$ of shape $(m, n)$?",
      options: [
        "It extracts column $j$ from the matrix, producing a vector of length $m$",
        "It sums along the rows (axis 0), producing a vector of length $n$ where each entry is $\\sum_i A_{ij}$",
        "It transposes the matrix and then flattens it into a vector of length $n$",
        "It computes the column-wise maximum, producing a vector of length $n$"
      ],
      correct: 1,
      explanation: "Index $i$ appears in the input but not the output, so it is summed over. Index $j$ survives. The result is a vector of length $n$: $y_j = \\sum_i A_{ij}$. This sums each column of $A$, which is equivalent to summing along axis 0 (the row axis)."
    },
    {
      type: "info",
      title: "Dot Product, Element-wise Product, and Outer Product",
      content: "Einsum unifies several vector operations under one notation.\n\n**Dot product** — both indices are contracted:\n$$\\texttt{'i,i->'} \\quad \\Rightarrow \\quad \\sum_i a_i \\, b_i$$\n\nThe shared index $i$ is absent from the output (a scalar), so it is summed. This gives $\\mathbf{a} \\cdot \\mathbf{b}$.\n\n**Element-wise (Hadamard) product** — the shared index survives:\n$$\\texttt{'i,i->i'} \\quad \\Rightarrow \\quad c_i = a_i \\, b_i$$\n\nIndex $i$ appears in the output, so no summation occurs — we just multiply corresponding entries.\n\n**Outer product** — no shared indices at all:\n$$\\texttt{'i,j->ij'} \\quad \\Rightarrow \\quad C_{ij} = a_i \\, b_j$$\n\nBoth indices survive in the output, so every pair is multiplied. A vector of length $m$ and a vector of length $n$ produce an $m \\times n$ matrix.\n\nThe pattern is always the same: shared indices that vanish from the output get summed, everything else is kept."
    },
    {
      type: "mc",
      question: "Two vectors $\\mathbf{a}$ and $\\mathbf{b}$ each have length 128. Which einsum expression produces a $(128, 128)$ matrix where entry $(i, j)$ equals $a_i \\cdot b_j$?",
      options: [
        "`'i,i->ii'` — repeating the index creates a matrix from element-wise pairing",
        "`'i,i->'` — summing over the shared index yields the full pairwise product table",
        "`'i,j->ji'` — transposing the index order produces $b_j \\cdot a_i$ in a matrix",
        "`'i,j->ij'` — distinct indices with no contraction gives the outer product"
      ],
      correct: 3,
      explanation: "The outer product needs two distinct indices ($i$ for $\\mathbf{a}$, $j$ for $\\mathbf{b}$) that both survive in the output. `'i,j->ij'` gives $C_{ij} = a_i b_j$, which is the standard outer product. Option C (`'i,j->ji'`) would give $C_{ji} = a_i b_j$, which is the transposed outer product — not what we want since entry $(i,j)$ should be $a_i b_j$, not $a_j b_i$."
    },
    {
      type: "info",
      title: "Matrix Multiply as Einsum",
      content: "Matrix multiplication is the most common einsum pattern in deep learning. For matrices $A \\in \\mathbb{R}^{m \\times p}$ and $B \\in \\mathbb{R}^{p \\times n}$:\n\n$$\\texttt{'ij,jk->ik'} \\quad \\Rightarrow \\quad C_{ik} = \\sum_j A_{ij} \\, B_{jk}$$\n\nThe index $j$ is the **contraction index** — it appears in both inputs and is absent from the output. The dimensions must match along $j$: $A$ has $p$ columns and $B$ has $p$ rows.\n\nYou can also express related operations:\n- **$A^T B$**: `'ji,jk->ik'` — $j$ is the first axis of $A$ (rows), contracted with $j$ in $B$\n- **$A B^T$**: `'ij,kj->ik'` — $j$ is the second axis of both, contracted away\n\nThe key insight: **which index is shared and eliminated determines which product you get**. The positions of indices in the input strings tell einsum how the axes align."
    },
    {
      type: "mc",
      question: "You have weight matrix $W$ of shape $(d_{\\text{in}}, d_{\\text{out}})$ and input $X$ of shape $(d_{\\text{in}},)$. Which einsum computes $W^T X$ (a vector of shape $(d_{\\text{out}},)$)?",
      options: [
        "`'ij,j->i'` — contracts over the second axis of $W$ and the vector index",
        "`'ij,i->j'` — contracts over the first axis of $W$ (the $d_{\\text{in}}$ axis) and the vector index",
        "`'ji,j->i'` — transposes $W$ first, then contracts over the leading axis",
        "`'j,ij->i'` — places the vector first to indicate it is the left operand in the product"
      ],
      correct: 1,
      explanation: "$W$ has shape $(d_{\\text{in}}, d_{\\text{out}})$, indexed as $W_{ij}$ where $i$ ranges over $d_{\\text{in}}$ and $j$ over $d_{\\text{out}}$. $X$ has shape $(d_{\\text{in}},)$ indexed by $i$. We want $y_j = \\sum_i W_{ij} X_i = (W^T X)_j$. The expression `'ij,i->j'` contracts over $i$ (the $d_{\\text{in}}$ axis) and outputs index $j$ (the $d_{\\text{out}}$ axis). This correctly computes $W^T X$."
    },
    {
      type: "info",
      title: "Batched Operations",
      content: "Deep learning operates on batches. Einsum handles this naturally by adding a **batch index** that appears in all inputs and the output — so it is never contracted.\n\n**Batched matrix multiply** — a batch of $B$ matrix multiplications at once:\n$$\\texttt{'bij,bjk->bik'} \\quad \\Rightarrow \\quad C_{bik} = \\sum_j A_{bij} \\, B_{bjk}$$\n\nThe index $b$ (batch) survives in the output, so each batch element is processed independently. Only $j$ is contracted.\n\nThis is equivalent to:\n```\nfor each b:\n    C[b] = A[b] @ B[b]\n```\n\nbut einsum computes it in one fused operation. You can have **multiple batch dimensions** — for example, `'bhij,bhjk->bhik'` has both a batch dimension $b$ and a head dimension $h$, both preserved in the output.\n\nThe rule is unchanged: indices in the output are free (kept), indices only in inputs are contracted (summed)."
    },
    {
      type: "mc",
      question: "You have tensors $A$ of shape $(32, 8, 64, 64)$ and $B$ of shape $(32, 8, 64, 128)$ and write `'bhij,bhjk->bhik'`. What is the shape of the output and which index is contracted?",
      options: [
        "Output shape $(32, 8, 64, 128)$; index $j$ (size 64) is contracted because it is the only index absent from the output",
        "Output shape $(32, 8, 64, 64)$; index $k$ (size 128) is contracted because it only appears in the second input",
        "Output shape $(32, 8, 128, 64)$; index $i$ is contracted because it represents the inner dimension",
        "Output shape $(8, 32, 64, 128)$; indices $b$ and $h$ are swapped and $j$ is contracted"
      ],
      correct: 0,
      explanation: "The output string is 'bhik'. Indices $b=32$, $h=8$, $i=64$, $k=128$ all survive, giving shape $(32, 8, 64, 128)$. Index $j$ appears in both inputs (size 64 in both) but is absent from the output, so $j$ is the contracted (summed) index. This is a batched matrix multiply across the $b$ and $h$ dimensions."
    },
    {
      type: "info",
      title: "Multi-Head Attention Einsums",
      content: "Multi-head attention involves two key einsum patterns. Given queries $Q$, keys $K$, and values $V$ each of shape $(B, H, S, D)$ — batch, heads, sequence length, head dimension:\n\n**Attention scores** ($QK^T$ per head):\n$$\\texttt{'bhid,bhjd->bhij'} \\quad \\Rightarrow \\quad \\text{scores}_{bhij} = \\sum_d Q_{bhid} \\, K_{bhjd}$$\n\nIndices $b, h$ are batch/head (preserved). Index $d$ (head dimension) is contracted. Indices $i, j$ are query and key positions — the output is an $(S \\times S)$ attention matrix per head.\n\n**Weighted value aggregation** (after softmax):\n$$\\texttt{'bhij,bhjd->bhid'} \\quad \\Rightarrow \\quad \\text{out}_{bhid} = \\sum_j \\text{attn}_{bhij} \\, V_{bhjd}$$\n\nNow $j$ (key/value position) is contracted — we sum over all positions weighted by attention. The result has shape $(B, H, S, D)$: each query position gets a weighted combination of value vectors.\n\nNotice how reading the indices tells you exactly what is happening: which dimensions are batch-like, which are contracted, and what shape the output has."
    },
    {
      type: "mc",
      question: "In the attention score computation `'bhid,bhjd->bhij'`, what would happen if you accidentally wrote `'bhid,bhjd->bhdij'` instead?",
      options: [
        "It would produce identical results but with an extra redundant dimension of size 1",
        "It would fail because $d$ cannot appear in the output — it must be contracted for attention to work",
        "It would produce a 5D tensor of shape $(B, H, D, S, S)$ with no contraction, giving pairwise products for each head-dimension element separately",
        "It would transpose the attention matrix, computing $K Q^T$ instead of $Q K^T$"
      ],
      correct: 2,
      explanation: "If $d$ appears in the output, it is no longer contracted (summed over). All indices $b, h, d, i, j$ survive, producing a 5D tensor of shape $(B, H, D, S, S)$ where each entry is $Q_{bhid} \\cdot K_{bhjd}$ — no summation over $d$. This gives the element-wise product for each head-dimension component separately, rather than the dot product across $d$ that attention requires."
    },
    {
      type: "info",
      title: "Reading Unfamiliar Einsum Expressions",
      content: "When you encounter an unfamiliar einsum pattern, decode it in three steps:\n\n**Step 1: Identify free indices** — those that appear in the output. These are preserved dimensions (batch, spatial, etc.).\n\n**Step 2: Identify contracted indices** — those in inputs but absent from the output. These are summed over. The dimension sizes must match between inputs sharing a contracted index.\n\n**Step 3: Interpret the operation** — free indices define the output shape; contracted indices define what is being \"dotted\" together.\n\nFor example, consider `'bnid,bnhd->bnhi'`:\n- Free indices: $b$ (batch), $n$ (some spatial dim), $h$ (heads), $i$ (position) — all in output\n- Contracted: $d$ — appears in both inputs, absent from output\n- Interpretation: for each $(b, n)$, compute a dot product over $d$ between a tensor indexed by $(i, d)$ and one indexed by $(h, d)$, producing an $(h, i)$ matrix. This is a batched $QK^T$-like operation with an extra spatial dimension $n$.\n\nThe beauty of einsum is that it makes the data flow explicit — there is no ambiguity about which axes align or contract."
    },
    {
      type: "mc",
      question: "You encounter the einsum `'bse,bte->bst'` where the first tensor has shape $(B, S, E)$ and the second has shape $(B, T, E)$. What does this compute?",
      options: [
        "A batched element-wise product between two sequence representations, producing shape $(B, S, T)$",
        "A bilinear projection that maps two sequences into a shared space of dimension $E$",
        "A concatenation of two sequence tensors along the embedding axis, producing shape $(B, S+T, E)$",
        "A batched pairwise similarity matrix of shape $(B, S, T)$ by contracting over the embedding dimension $e$"
      ],
      correct: 3,
      explanation: "Free indices are $b, s, t$ (all in the output 'bst'). The contracted index is $e$ (embedding dimension) — it appears in both inputs but not in the output. So for each batch element, we compute $Y_{bst} = \\sum_e X_{bse} \\cdot Z_{bte}$: a dot product over embeddings between every pair of positions $(s, t)$. This produces a $(B, S, T)$ pairwise similarity matrix — the same structure as attention logits before scaling and softmax."
    },
  ]
};
