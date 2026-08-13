# SPDX-License-Identifier: Apache-2.0
"""Roll out the B (project) notebooks for every chapter, plus the Ch7 a/b pair.

Each B notebook runs the REAL project step on the SHARED store/artifacts and
inspects the result, so running the chapters in order accumulates the complete
RAG by the capstone. A (inline) notebooks are the renamed per-chapter notebooks;
Ch6 a/b come from build_nb_06_doe_enrichment.py; Ch7 a/b are emitted here.

CPU-only authoring: cells are emitted, not executed. Run:
    python scripts/build_book_notebooks.py
"""
from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
# notebooks/ is a sibling of code/ (this file lives in code/scripts/), so go up
# twice: code/scripts -> code -> beyond-chunk-and-pray, then into notebooks/.
NBDIR = os.path.join(HERE, "..", "..", "notebooks")

BOOT = (
    'import os, sys\n'
    'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "")\n'
    'sys.path.insert(0, KNOWLYTIX_SRC)\n'
    'REPO = os.path.join(os.path.dirname(os.getcwd()), "code") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()\n'
    'sys.path.insert(0, os.path.join(REPO, "scripts"))'
)
# Build the shared capstone pipeline on the real store (accept gate open here;
# Ch14 calibrates it). Used by every retrieval/answer chapter's B notebook.
PIPE = (
    'import capstone_pipeline as cp\n'
    'store = cp.load_store(os.path.join(REPO, "data", "gms_annual_report_store"))\n'
    'pipe = cp.build_pipeline(store, cp.make_qwen(), accept_threshold=0.0)'
)
RUN = ('import subprocess, sys\n'
       'def run(*a): subprocess.run([sys.executable, os.path.join(REPO, "scripts", a[0]), *a[1:]], check=True)\n')


def _write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
                for k, s in cells]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}}
    with open(os.path.abspath(os.path.join(NBDIR, name)), "w") as fh:
        nbf.write(nb, fh)
    print("wrote", name, f"({len(cells)} cells)")


def _q(question, show="sources"):
    body = {
        "sources": f'a = pipe.query({question!r})\nprint("decision:", a.decision)\n'
                   'for f in a.sources[:5]: print(" ", (f.head, f.relation, f.tail), "@", f.location)',
        "answer": f'a = pipe.query({question!r})\nprint("decision:", a.decision, "| answer:", a.answer)',
    }[show]
    return body


# (new_num, slug, title, [B cells]) -- B runs the real step on shared artifacts.
B_CHAPTERS = [
    (1, "why_chunk_and_pray_fails", "Why chunk and pray fails",
     [("md", "Run the real baseline retriever and watch a chunk return many "
             "co-occurring numbers (project: `baseline_rag`)."),
      ("code", BOOT),
      ("code", 'import baseline_rag\n'
               'b = baseline_rag.BaselineRAG(open(os.path.join(REPO,"data","annual_report.md")).read(),\n'
               '                             lambda p: "")\n'
               'for c, s in b.retrieve("What was total revenue?", k=3):\n'
               '    nums = [t for t in c.text.replace("|"," ").split() if t.replace(".","",1).isdigit()]\n'
               '    print(f"cos={s:.3f} {c.location} numbers={nums[:8]}")')]),
    (2, "geometric_memory", "Geometric memory",
     [("md", "Call the six primitives on the real store."),
      ("code", BOOT),
      ("code", 'import capstone_pipeline as cp\n'
               'store = cp.load_store(os.path.join(REPO, "data", "gms_annual_report_store"))\n'
               'print("enm:", store.lookup_enm("segment_performance", "Cloud Platform/Technology/Revenue"))\n'
               'print("triple:", store.query_triples(head="cloud platform", relation="has_division"))')]),
    (3, "provenance", "Provenance",
     [("md", "Resolve a retrieved triple to its source span on the real store."),
      ("code", BOOT), ("code", PIPE),
      ("code", 'a = pipe.query("What is Cloud Platform revenue?")\n'
               'for f in a.sources: print((f.head, f.relation, f.tail), "@", f.location)')]),
    (4, "document_to_graph", "From document to graph",
     [("md", "Run the project build: GeodeLoop -> trained store + ENM + coverage."),
      ("code", BOOT), ("code", RUN + 'run("build_store.py")'),
      ("code", 'import json\n'
               'print(json.load(open(os.path.join(REPO,"data","gms_annual_report_store","coverage_report.json")))["coverage_ratio"])')]),
    (5, "geode_self_correction", "GEODE self-correction",
     [("md", "GEODE corrected the graph during the Ch4 build; inspect the coverage "
             "and the corrected triples on the real store."),
      ("code", BOOT),
      ("code", 'import capstone_pipeline as cp\n'
               'store = cp.load_store(os.path.join(REPO, "data", "gms_annual_report_store"))\n'
               'print("triples:", len(store.query_triples()))')]),
    (8, "triple_mediated_retrieval", "Triple-mediated retrieval",
     [("md", "Triple-mediated retrieval on the real store + tuned encoders."),
      ("code", BOOT), ("code", PIPE), ("code", _q("What is Cloud Platform revenue?"))]),
    (9, "binding", "Binding",
     [("md", "Bind a paraphrased query to the graph through the tuned v-encoder."),
      ("code", BOOT), ("code", PIPE),
      ("code", 'print(pipe.extract("How much did the cloud segment sell?").bound_triples)')]),
    (10, "binder_bakeoff", "Choosing a binder: a bake-off",
     [("md", "The work of the bake-off is **building** the binders; comparing them is a "
             "much smaller step. This notebook builds all four over the annual-report "
             "store --- it generates training samples from the store, fine-tunes the "
             "compiler (arm B), indexes those samples for the frozen few-shot binder "
             "(arm C), and calibrates the patchable alias table (arm D) --- and only "
             "then reads the persisted crossover. The build cells need the GPU and the "
             "licensed backend; run them locally (execution is off at render time)."),
      ("md", "**Setup.** Load the store and the base LM through the installed package."),
      ("code",
       "from forgeloop import data_path, ensure_artifacts\n"
       "from forgeloop.rag import load_store\n"
       "from knowlytix.knowledge.llm_backend import LocalTransformersBackend\n"
       "from knowlytix.knowledge.rag.compiler import (build_compiler_dataset, train_compiler,\n"
       "                                              CompilerSFTConfig, QWEN_4B)\n"
       "ensure_artifacts()          # fetch the store + adapters if missing (portable)\n"
       "store = load_store()        # forgeloop resolves the store dir -- no hardcoded path\n"
       "llm = LocalTransformersBackend(QWEN_4B)   # rephraser for datagen + base for SFT"),
      ("md", "**Arm B, step 1 --- generate samples (a design of experiments).** "
             "`build_compiler_dataset` runs three knowlytix stages: graph generators "
             "mine base questions (single-hop, two-hop, and relation-absent probes), "
             "each carrying its gold hop chain as the training target; "
             "`DesignMatrix.from_catalog` draws a space-filling **Sobol** design over "
             "the ~20 presentation factors of the DoE (clarity, length, expertise, "
             "paraphrase depth, ...); and `QuestionRephraser` realizes each design row, "
             "rewriting the base question to those factor levels while preserving the "
             "target chain. A held-out factor level and a fraction of facts become the "
             "eval splits, so the design defines what the model is tested on."),
      ("code",
       "splits = build_compiler_dataset(\n"
       "    store, llm, group='comprehensive', variants_per_base=12,\n"
       "    heldout_levels={'clarity': 'Misleading'}, heldout_fact_frac=0.15, seed=42)\n"
       "print({k: len(v) for k, v in splits.items()})\n"
       "print('one sample:', splits['train'][0])"),
      ("md", "**Arm B, step 2 --- fine-tune the SLM compiler** on the samples with a "
             "low-rank adapter, written into the store's `query_compiler/`."),
      ("code",
       "compiler_dir = train_compiler(\n"
       "    splits['train'], out_dir=str(data_path('gms_annual_report_store', 'query_compiler')),\n"
       "    config=CompilerSFTConfig(base_model=QWEN_4B, lora_r=16, lora_alpha=32, epochs=3))\n"
       "print('compiler adapter:', compiler_dir)"),
      ("md", "**Arm C --- index the same samples** in the tuned encoder's space (no "
             "training); the k nearest are shown to a frozen model at inference."),
      ("code",
       "from knowlytix.embedding import FineTunedEmbedding\n"
       "from knowlytix.knowledge.rag.bakeoff import ExemplarIndex\n"
       "v_encoder = FineTunedEmbedding.load(str(data_path('gms_annual_report_store', 'tuned_encoder')))\n"
       "index = ExemplarIndex.from_rows(splits['train'], v_encoder.encode)\n"
       "print('exemplars indexed:', len(index))"),
      ("md", "**Arm D --- build a patchable alias table** for the head entity, with a "
             "calibrated encoder-nearest fallback; a mis-binding is fixed by one edit."),
      ("code",
       "from knowlytix.knowledge.rag.bakeoff import AliasTableResolver, calibrate_alias_resolver\n"
       "from knowlytix.knowledge.rag.compiler.walk import StoreChainWalker\n"
       "entities = StoreChainWalker(store).entities\n"
       "tau = calibrate_alias_resolver(entities, v_encoder.encode, far_ceiling=0.05)['tau']\n"
       "resolver = AliasTableResolver.from_store(store, encoder=v_encoder.encode, tau=tau)\n"
       "resolver.add_alias('CP', 'cloud platform')   # a patch is one row, not a retrain\n"
       "print('alias entries:', len(resolver.table), '| fallback tau:', round(tau, 3))"),
      ("md", "**The comparison, in one step.** With the binders built, the crossover and "
             "the G4 verdict are read from the persisted run."),
      ("code",
       "import json\n"
       "report = json.load(open(data_path('enrichment', 'bakeoff_ABCD.json')))\n"
       "decision = json.load(open(data_path('enrichment', 'bakeoff_decision.json')))\n"
       "for name, e in report['arms'].items():\n"
       "    o = e['overall']\n"
       "    print(f\"{name:12} acc={o['accuracy']:.3f} mis_bind={o['mis_bind_rate']:.3f} \"\n"
       "          f\"holdout={e['holdout']['accuracy']:.3f} patch={e['patch_cost']}\")\n"
       "print('decision:', decision['recommended'], '| recused:', decision['recused'])")]),
    (11, "answering_through_the_gms", "Answering through the GMS",
     [("md", "Multi-hop answer through the real graph."),
      ("code", BOOT), ("code", PIPE),
      ("code", _q("Which region runs the division that contains Cloud Platform?", "answer"))]),
    (12, "grounded_synthesis", "Grounded synthesis",
     [("md", "Synthesize a grounded answer from retrieved facts (real Qwen)."),
      ("code", BOOT), ("code", PIPE), ("code", _q("What was net income in FY2025?", "answer"))]),
    (13, "self_verification", "Self-verification",
     [("md", "The GMS catches a confident-wrong number (u-space contradiction)."),
      ("code", BOOT), ("code", PIPE),
      ("code", 'a = pipe.query("What is Cloud Platform revenue?")\n'
               'print("decision:", a.decision, "| verified:", a.verified)')]),
    (14, "abstention_and_coverage", "Abstention and coverage",
     [("md", "Abstain on a prose blind spot; coverage_report names the blind spots."),
      ("code", BOOT), ("code", PIPE),
      ("code", 'from knowlytix.knowledge.rag import coverage_report\n'
               'print(pipe.query("What is management\\u2019s outlook for fiscal 2026?").decision)\n'
               'print(round(coverage_report(store).coverage_ratio, 2))')]),
    (15, "calibration", "Calibration",
     [("md", "Calibrate the accept gate from the cohort (project step)."),
      ("code", BOOT), ("code", RUN + 'run("calibrate_accept_gate.py")'),
      ("code", 'import json\n'
               'print(json.load(open(os.path.join(REPO,"data","gms_annual_report_store","rag_gate_calibration.json"))))')]),
    (16, "evaluation", "Evaluating the RAG",
     [("md", "Run the DoE evaluation and the GEODE-vs-baseline comparison (project)."),
      ("code", BOOT), ("code", RUN + 'run("rag_doe_compare.py", "--limit", "150", "--k", "3")'),
      ("code", 'import json\n'
               'd = json.load(open(os.path.join(REPO,"data","enrichment","rag_doe_compare.json")))\n'
               'for m in ["precision_at_k","recall_at_k","correctness","completeness","abstention_rate"]:\n'
               '    print(f"{m:16}{d[\'geode\'][m]:>8.3f}{d[\'baseline\'][m]:>10.3f}")')]),
    (17, "pluggable_llms_and_dense_fallback", "Pluggable LLMs",
     [("md", "Same pipeline, swap the backend; dense fallback stays off."),
      ("code", BOOT), ("code", PIPE), ("code", _q("What was total revenue?", "answer"))]),
    (18, "external_persistence_kal", "Persisting to KAL",
     [("md", "Persist the verified graph to KAL's offline mock and round-trip."),
      ("code", BOOT),
      ("code", 'import torch\n'
               'import capstone_pipeline as cp\n'
               'from knowlytix.kal.adapters import MockKnowledgeAdapter\n'
               'from knowlytix.knowledge.rag.kal_sink import persist_store_to_kal_sync, store_to_kal_triples\n'
               'store = cp.load_store(os.path.join(REPO, "data", "gms_annual_report_store"), dev=torch.device("cpu"))\n'
               'n = persist_store_to_kal_sync(MockKnowledgeAdapter("capstone"), store,\n'
               '        tenant_id="northwind", source="annual_report.md", confidence=1.0)\n'
               'print("persisted", n, "of", len(store_to_kal_triples(store, source="annual_report.md")))')]),
    (19, "capstone_summary", "Capstone: the complete pipeline + verdict",
     [("md", "The complete RAG, assembled, and the head-to-head verdict the book "
             "concludes on (project: the full comparison)."),
      ("code", BOOT), ("code", RUN + 'run("rag_doe_compare.py", "--limit", "150", "--k", "3")'),
      ("code", 'import json\n'
               'd = json.load(open(os.path.join(REPO,"data","enrichment","rag_doe_compare.json")))\n'
               'g, b = d["geode"], d["baseline"]\n'
               'print("GEODE wins precision %.2f vs %.2f, correctness %.2f vs %.2f; only GEODE abstains %.2f vs %.2f"\n'
               '      % (g["precision_at_k"], b["precision_at_k"], g["correctness"], b["correctness"],\n'
               '         g["abstention_rate"], b["abstention_rate"]))')]),
]

# Ch7 a/b (embedding SFT) -- new chapter.
CH7_A = [
    ("md", "# Ch7 (A, inline) - Tuning the Encoders: Embedding SFT\n\n"
           "v-space pulls a question to the right attribute; u-space separates a "
           "true claim from a conflicting-value one. Both tune a low-rank adapter "
           "over a frozen base, on the Ch6 generated data."),
    ("code", BOOT),
    ("code", 'from knowlytix.embedding import EmbeddingSFTConfig, finetune_embedding\n'
             'import torch\n'
             'import capstone_pipeline as cp\n'
             'store = cp.load_store(os.path.join(REPO, "data", "gms_annual_report_store"))\n'
             'd_v = store.model.cfg.d_v\n'
             'v_ft = finetune_embedding(os.path.join(REPO,"data","enrichment","embedding_sft.jsonl"),\n'
             '    EmbeddingSFTConfig(rank=8, mode="full", out_dim=d_v, objective="prototype"),\n'
             '    text_col="text", label_col="label")\n'
             'print("v-space val_accuracy =", round(v_ft.val_accuracy, 3))'),
    ("md", "u-space: genuine conflicting-value pairs, full mode (a rotation cannot move tension)."),
    ("code", 'from finetune_encoders import _contradiction_pairs, _fit_contradiction_pairs, _uspace_tension\n'
             'pos, neg = _contradiction_pairs(store)\n'
             'u_ft = _fit_contradiction_pairs(pos, neg, EmbeddingSFTConfig(\n'
             '    rank=32, mode="full", objective="contradiction",\n'
             '    encoder="sentence-transformers/nli-mpnet-base-v2", out_dim=d_v, epochs=400))\n'
             'print("consistent:", round(_uspace_tension(u_ft, "Retail\\u2019s headcount was 520.",\n'
             '      "The headcount of Retail is 520."), 3))\n'
             'print("contradictory:", round(_uspace_tension(u_ft, "Retail\\u2019s headcount was 520.",\n'
             '      "Retail\\u2019s headcount was 210."), 3))'),
]
CH7_B = [
    ("md", "# Ch7 (B, project) - run the real SFT step\n\n"
           "Run `scripts/finetune_encoders.py`: v/u SFT on the Ch6 data + relevance "
           "calibration, writing tuned_encoder/, contradiction_encoder/ and "
           "relevance_calibration.json beside the store."),
    ("code", BOOT),
    ("code", RUN + 'run("finetune_encoders.py")'),
    ("code", 'import os\n'
             'S = os.path.join(REPO, "data", "gms_annual_report_store")\n'
             'for d in ["tuned_encoder", "contradiction_encoder"]:\n'
             '    assert os.path.isdir(os.path.join(S, d)), d\n'
             'print("OK: tuned encoders + relevance_calibration.json ready for retrieval (Ch8+)")'),
]


# Ch15 A (inline DoE evaluation) -- replaces the stale renamed eval notebook.
CH15_A = [
    ("md", "# Ch15 (A, inline) - Evaluating the RAG with a Designed Experiment\n\n"
           "The DoE cohort (Ch6) is the test set, the GMS is the oracle. We measure "
           "precision/recall@k, calibrated correctness, completeness, and attribute "
           "failures to the presentation factors."),
    ("code", BOOT), ("code", PIPE),
    ("code", 'import json, re\n'
             'from knowlytix.harness.testing.hallucination import HallucinationOracle\n'
             'from knowlytix.harness.testing.completeness import CompletenessEvaluator\n'
             'oracle, comp = HallucinationOracle(store=store), CompletenessEvaluator(store)\n'
             'cohort = json.load(open(os.path.join(REPO,"data","enrichment","rag_cohort.json")))\n'
             'def gtv(e): return str(e[1]) if isinstance(e, list) else str(e)\n'
             'def golden(c):\n'
             '    g = gtv(c["expected_answer"])\n'
             '    return [(h,r,str(t)) for h,r,t in store.triples if str(t)==g][:1]'),
    ("code", 'P=R=C=CO=AB=N=NA=0\n'
             'for c in cohort:\n'
             '    a = pipe.query(c["question"]); g = gtv(c["expected_answer"]); N+=1\n'
             '    top = a.sources[:3]; hit = [f for f in top if str(f.tail)==g]\n'
             '    P += len(hit)/max(1,len(top)); R += 1.0 if hit else 0.0\n'
             '    if a.decision != "accept": AB += 1; continue\n'
             '    NA += 1; gold = golden(c)\n'
             '    if gold:\n'
             '        h,r,_ = gold[0]; m = re.findall(r"\\d[\\d,]*\\.?\\d*", (a.answer or "").replace(",",""))\n'
             '        CO += 1.0 if (m and oracle.assess_claim(h,r,m[0]).passed) else 0.0\n'
             '        C += comp.evaluate(a.answer or "", [g], {"head": h}).score\n'
             'print(f"precision@3={P/N:.3f} recall@3={R/N:.3f} correctness={CO/max(1,NA):.3f} "\n'
             '      f"completeness={C/max(1,NA):.3f} abstention={AB/N:.3f}")'),
]

# Ch18 A (inline capstone) -- the complete pipeline run + the verdict.
CH18_A = [
    ("md", "# Ch18 (A, inline) - the complete pipeline, and the verdict\n\n"
           "Everything assembled: one pipeline over the store built and tuned across "
           "Ch4-7, answering through the graph with provenance, abstaining on prose, "
           "and the head-to-head verdict from Ch15."),
    ("code", BOOT), ("code", PIPE),
    ("code", 'for q in ["What is Cloud Platform revenue?",\n'
             '          "Which region runs the division that contains Cloud Platform?",\n'
             '          "What is management\\u2019s outlook for fiscal 2026?"]:\n'
             '    a = pipe.query(q)\n'
             '    print(f"{a.decision:8} {q[:52]:52} -> {a.answer[:40]}")'),
    ("md", "The verdict the book concludes on (from the Ch15 comparison)."),
    ("code", 'import json\n'
             '_cmp = os.path.join(REPO, "data", "enrichment", "rag_doe_compare.json")\n'
             'if not os.path.exists(_cmp):\n'
             '    print("rag_doe_compare.json not found \\u2014 run scripts/rag_doe_compare.py (GPU + Qwen) to build it.")\n'
             'else:\n'
             '    d = json.load(open(_cmp))\n'
             '    g, b = d["geode"], d["baseline"]\n'
             '    print(f"precision  GEODE {g[\'precision_at_k\']:.2f} vs baseline {b[\'precision_at_k\']:.2f}")\n'
             '    print(f"correctness GEODE {g[\'correctness\']:.2f} vs baseline {b[\'correctness\']:.2f}")\n'
             '    print(f"abstention GEODE {g[\'abstention_rate\']:.2f} vs baseline {b[\'abstention_rate\']:.2f}")'),
]


def build() -> None:
    for num, slug, _title, cells in B_CHAPTERS:
        _write(f"{num:02d}_{slug}_b_project.ipynb", cells)
    _write("07_embedding_sft_a_inline.ipynb", CH7_A)
    _write("07_embedding_sft_b_project.ipynb", CH7_B)
    _write("16_evaluation_a_inline.ipynb", CH15_A)
    _write("19_capstone_summary_a_inline.ipynb", CH18_A)


if __name__ == "__main__":
    build()
