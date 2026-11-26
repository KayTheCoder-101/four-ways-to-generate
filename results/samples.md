# RAG examples

Passages are shortened. "Relevant" means the abstract id is in the question's gold `relevant_ids`. All five come from `results/rag_answers.jsonl` (first 100 test questions).

## 1. Hallucination: capmatinib (E.6)

**Question (qid 4213):** Is capmatinib effective for glioblastoma?

**Answer:** Yes, capmatinib is effective for glioblastoma. [3]

**Reference:** No. The combination of capmatinib and buparlisib showed no clear activity in recurrent PTEN-deficient glioblastoma.

**Cited passage [3]:** "Capmatinib (Tabrecta™) is an oral, small molecule mesenchymal-epithelial transition (MET) inhibitor being developed by Novartis Oncology … for the treatment of lung cancer …"

**What went wrong:** Passage [3] only says capmatinib is a MET inhibitor being developed for lung cancer. Nothing in it says it works for glioblastoma. The gold abstract (31776899) was not retrieved, not even in the top 10 abstracts, so this is also a retrieval failure, and the other passages were general glioblastoma text. The model joined "capmatinib" and "glioblastoma" and confidently said yes. The citation is also wrong: 0 of 1 citations point to a relevant abstract.

## 2. Retrieval failure: F105-P

**Question (qid 4456):** What is F105-P?

**Answer:** I don't know.

**Reference:** F105-P is a protamine-antibody fusion protein designed to deliver siRNA to HIV-infected or envelope-transfected cells.

**What went wrong:** None of the five passages is about F105-P. They cover ACVR1 mutations, a driver-gene database, an NPC1 probe called FP-5, an iron-export protein and a STAT5 reporter assay. The embedding model most likely matched on the letter-and-number pattern of the name, not its meaning. The gold abstract (15908939) was not in the top 10 abstracts (Hit@10 = 0). Here the generator did the right thing, since there was nothing to build on.

## 3. "I don't know" that should have been an answer: dasatinib

**Question (qid 3609):** Which disease is Dasatinib used to treat?

**Answer:** The passage does not explicitly state what disease Dasatinib is used to treat. Therefore, the answer is I don't know.

**Reference:** Patients with chronic myeloid leukemia.

**What went wrong:** The passages clearly describe dasatinib treatment in patients with chronic-phase chronic myeloid leukemia (passage [4]) and, with blinatumomab, in Ph-positive acute lymphoblastic leukemia (passage [2]). They mention it in passing, never as a definition, so the 1.5B model refused. This is the opposite failure to example 2: the evidence was there and the model was too cautious. It counts toward the 16% "I don't know" rate without being a retrieval problem.

## 4. Good answer: tirabrutinib

**Question (qid 4628):** Is tirabrutinib effective for lymphoma?

**Answer:** Yes. It cites passages [2], [3] and [4] for approval in Japan for relapsed or refractory primary CNS lymphoma, for a phase I/II trial with a complete response, and for the first human evidence of effectiveness. It concludes that tirabrutinib "shows promise in treating lymphoma".

**Reference:** Yes, tirabrutinib appears to be effective for lymphoma. It was approved in Japan for recurrent or refractory primary central nervous system lymphoma.

**Why it works:** Four of the five retrieved abstracts are about tirabrutinib, and all four of the answer's citations ([2], [2], [3], [4]) point to relevant abstracts. The answer matches the reference and is grounded in the passages. Its citations also check out in detail: the March 2020 Japanese approval for relapsed or refractory primary CNS lymphoma is stated in both [2] and [4], and the answer cites [2] for it.

**Cited passage [2]:** "In March 2020, oral tirabrutinib was approved in Japan for the treatment of recurrent or refractory primary central nervous system lymphoma."

## 5. Good answer: selpercatinib

**Question (qid 3867):** Which cancers can be treated with Selpercatinib?

**Answer:** Based on passage [1]: cancers with RET alterations, specifically RET fusion-positive non-small-cell lung cancer, RET fusion-positive thyroid cancer and RET-mutant medullary thyroid cancer, supported by the phase I/II LIBRETTO-001 trial and its FDA approval.

**Reference:** Recently approved by the US FDA for RET fusion-positive NSCLC, RET fusion-positive thyroid cancer and RET-mutant medullary thyroid cancer.

**Why it works:** Passage [1] contains the answer almost word for word. The model copied the right facts and cited the right passage. It shows the best case: a question with a clear, concentrated answer in the top-ranked chunk.
