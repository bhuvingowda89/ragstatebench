# RAGStateBench

Deterministic retrieval substrate for **RAGStateBench: Benchmarking State Convergence and Integrity Failures in Enterprise Retrieval-Augmented Generation**. This repository evaluates retrieval and context boundaries after knowledge and authorization transitions. It does not generate answers with an LLM.

## Reproduce the pilot

Python 3.10 or newer is required. On this machine, `python3` is Python 3.9, so use `python3.11`. From the repository root:

```bash
python3.11 -m pip install -e .
python3.11 -m pytest -q
ragstate validate-clean --campaign pilot --config configs/pilot.yaml
ragstate validate-controls --campaign pilot --config configs/pilot.yaml
ragstate run --campaign pilot --config configs/pilot.yaml
```

The bundled `configs/pilot.yaml` is JSON syntax, which is valid YAML 1.2. This keeps the CLI free of a YAML parser dependency. `validate-controls` runs the same campaign and checks positive and negative controls. `run` resumes completed scenarios when the manifest is unchanged. Change `output` or pass `--output` to isolate a new run. A changed manifest is rejected in an existing output directory.

The clean gate evaluates answerable, structured queries at CER@10 and exits nonzero below 0.90. The example config runs BM25 on CPU and needs no model download. For dense or hybrid retrieval, install `ragstatebench[dense]`, set `retriever` to `dense` or `hybrid`, and set `embedding_model_version` to an immutable Hugging Face revision for `sentence-transformers/all-MiniLM-L6-v2` or another compatible small model. Model loading happens only for a dense or hybrid campaign; unit tests inject a deterministic encoder. Document embeddings are batched and cached under `embedding_cache_dir` (default `.cache/ragstate`). The configured model revision is passed to the loader and recorded in every result.

For a small real CPU validation of both retrievers on the unchanged clean-query set:

```bash
python3.11 -m pip install -e '.[dense]'
PYTHONPATH=src python3.11 scripts/smoke_retrievers.py \
  --config configs/pilot.yaml \
  --model-revision acbb28c8aa70f5503c85d6b90e8cd65606993a20
```

The smoke runs two independent passes and reports the exact model revision, indexed corpus size, separate model load and cold document encoding times, dense and hybrid CER@10, median and p95 warm query latency, representative gold-evidence origins, and ranking stability. It creates a fresh temporary embedding cache for each pass, so encoding time is not confused with a disk-cache hit. It writes `results/dense-smoke/{summary.json,per_query.csv,per_query_run2.csv}`, the equivalent hybrid files, and `results/retrieval-smoke-comparison.csv`. Multi-document questions have one CSV row per required document; overlap counts require all gold documents for a query to be present. It does not launch the scenario campaign.

For the local offline MiniLM snapshot prepared in this workspace, use the existing CPU environment and avoid any downloads:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python scripts/smoke_retrievers.py \
  --config configs/pilot.yaml \
  --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --model-path /Users/nischithacn/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

## State and evaluation boundaries

Canonical `WorldLedger` state drives the query oracle and evaluation. Separate index and authorization ledgers receive scheduled events. Retrieval sees only index chunks. The indexed authorizer sees only its authorization replica. A4 uses live authorization; A5 additionally revalidates source document identity and content at context construction. No canonical gold evidence is passed to a retriever.

The six architecture labels have explicit pilot behavior:

| Baseline | Behavior |
| --- | --- |
| A0 | Eventual replica delivery and reusable cache |
| A1 | A0 plus periodic source and authorization sync (`sync_interval`) |
| A2 | Invalidate cached context after delivered events |
| A3 | Invalidate cached context after relevant delivered events |
| A4 | Live authorization check |
| A5 | Live authorization and source-version/content revalidation with event invalidation |

BM25, dense cosine retrieval, and weighted reciprocal rank fusion are the three retrieval families. Hybrid fusion preserves component ranks and scores. The generator creates six domains, multiple tenants and groups, near-duplicate policy documents, versioned facts, and hundreds of principals when configured at larger scale. Separate seeds control world, query, transition, and fault generation.

## Pilot artifacts

The output directory contains:

- `manifest.json`: deterministic configuration and scenario definitions.
- `status.json`: per-scenario completion or failure reason.
- `results.jsonl`: one full trace per evaluated query-state pair, including raw, filtered, and context hits, structured gold, state hashes, schedule, seeds, model metadata, latency, and available Git commit.
- `summary.csv`: flattened canonical analysis rows.
- `aggregate.csv`: grouped means by architecture, retriever, and fault family.

The pilot includes update, supersede, delete, grant, and revoke transitions; delay, partial, drop, out-of-order, stale cache, stale ACL, mixed-version, cross-principal, and compound faults; plus six negative controls. `GROUP_REMOVE` is a secondary implementation scenario. The control command verifies failure signals and checks that unrelated changes do not create benchmark failures.

## Metric conventions and limits

The runner records SCRS, CER@k, SRR, UCER, ARDR, MVCR, deleted evidence persistence, cross-tenant exposure, revocation exposure window, retrieval state convergence time, and stale event resurrection. The full trace retains raw candidates, filtered hits, and context hits separately. Candidate, post-filter, and context UCER fields localize authorization violations; primary UCER is context-level.

`CER@k` is `null` when no current evidence is required. CSV writes an empty field, and aggregate means exclude those rows. `invalid_evidence_absence_success` is an internal, non-headline measure: for empty gold it is 1 exactly when no chunk from a query-target logical document reaches context; it is `null` for answerable states. Other unrelated context chunks do not affect this absence measure, but can still affect SCRS and exposure metrics.

UCER counts policy denial only. For a current document, the evaluator checks canonical authorization. For a deleted version still recorded in the canonical ledger, it evaluates the active direct rule, tenant, and group policy using that version's metadata without reviving it. A deleted document with no independent denial contributes to DEP but not UCER; a simultaneous revoke can contribute to both. Unknown document IDs are outside the policy evaluation universe and do not increment UCER. SRR excludes deleted evidence in campaign evaluation, leaving deletion persistence to DEP. `T_RSC` is `null` if stable convergence is not observed by the configured horizon, so it is right-censored. REW is measured only through the observed horizon. Latency is a runtime measurement and is not deterministic.

The CPU sentence-transformer baseline requires a model download unless cached locally; the mock-encoder tests do not use the network. The real local-snapshot smoke is recorded in `results/dense-smoke/` and `results/hybrid-smoke/`. Larger-corpus quality, domain realism, and statistical analysis remain pilot questions. Stale answer-cache scenarios retain structured derived claims and their source chunks; they do not model generated prose. Final-output metrics require the later generation phase; no answer-level outcomes are inferred from retrieval traces.

## Scientific-pilot-v1 clean preflight and stop decision

The Phase 7 preflight uses a separate, six-domain corpus with 30 documents and 30 base questions per domain. Its configuration is frozen in `configs/scientific-pilot-v1-preflight.json`; it does not modify the earlier pilot corpus or results. Reproduce it with:

```bash
PYTHONPATH=src .venv-dense/bin/python -m ragstate.campaign.scientific_preflight \
  --config configs/scientific-pilot-v1-preflight.json \
  --output results/scientific-pilot-v1-preflight-new
```

The clean preflight triggered the specified stop condition before transition or mitigation campaigns: across 150 answerable questions, BM25 SCRS was 1.0, dense SCRS was 0.6067, and hybrid SCRS was 0.8733. Dense CER@10 was 0.6333 and hybrid CER@10 was 0.9133. Multi-document clean SCRS was 0.0 for dense and 0.5333 for hybrid. The query set, labels, top-k, and fusion settings were not changed after observing these results. `results/scientific-pilot-v1-preflight/` preserves the manifest, per-query table, and summary; no `results/scientific-pilot-v1/` fault campaign was launched.

## Phase 7A clean retrieval diagnostic

The immutable preflight was diagnosed at depths 5, 10, 20, and 50. The model screen uses exactly MiniLM, BGE small English v1.5, and E5 small v2 from local Hugging Face snapshot directories. The frozen selection rule is in `results/phase7a-diagnostic-v1/manifest.json`; it chooses the highest six-domain macro dense SCRS, then multi-document SCRS, macro CER@10, and lower CPU query latency. BGE uses its query instruction; E5 uses its documented query and passage prefixes. The benchmark corpus, gold labels, top-k 10, and RRF configuration were not changed.

From the repository root, with the three snapshots cached locally, reproduce the diagnostic offline into a **new** output directory containing a copy of the frozen manifest:

```bash
mkdir -p results/phase7a-reproduction
cp results/phase7a-diagnostic-v1/manifest.json results/phase7a-reproduction/manifest.json
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python -m ragstate.campaign.phase7a --output results/phase7a-reproduction
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python -m ragstate.campaign.phase7a_screen --output results/phase7a-reproduction
```

The screen refuses to overwrite a completed run and verifies the preflight artifact hashes before evaluation. `failure_diagnostics.csv`, `multi_document_diagnostic.csv`, `depth_sensitivity.csv`, `corpus_realism.json`, `dense_margin_summary.json`, `model_comparison.csv`, model-level query files, and six automatically generated SVG/PNG figure pairs are under the diagnostic directory. D1 failed because every answerable query contains a gold-only exact identifier. No transition-fault campaign was run; any corpus redesign must be versioned separately.

## Phase 7B versioned corpus validation

Corpus v2 is generated independently of the v1 scientific preflight. It has six domain-specific sets of policy topics, scoped variants across two context axes, separate default and approved-exception records, 420 stored versions for 360 logical records, and 216 structured questions. The six family counts per domain and clean gates are frozen in `configs/phase7b-corpus-v2.json`. Atomic gold claims identify the current concrete version, numeric value, and precedence of baseline versus exception evidence. Non-identifier questions are checked for gold-only lexical tokens; flagged construction errors are regenerated with fixed alternate templates or retained with a warning. The generator never reads retrieval scores.

The local E5 snapshot is required. From the repository root, create fresh directories with:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python -m ragstate.campaign.phase7b generate \
  --config configs/phase7b-corpus-v2.json \
  --data data/generated-v2-reproduction \
  --output results/phase7b-corpus-validation-v2-reproduction
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python -m ragstate.campaign.phase7b validate \
  --config configs/phase7b-corpus-v2.json \
  --data data/generated-v2-reproduction \
  --output results/phase7b-corpus-validation-v2-reproduction
```

Generation writes an immutable corpus manifest and document, principal, query, and leakage files before retrieval runs. Validation checks their hashes, scores BM25, the pinned E5 model, and the unchanged RRF hybrid at top-k 10, then repeats the index and query pass for rank determinism. Outputs include per-query and group CSVs, corpus and gate JSON summaries, and six SVG/PNG figure pairs. Completed outputs refuse overwriting.

The first v2 validation exposed a formatting bug: chunk construction collapses paragraph whitespace, while SCRS compared that chunk against source text with literal newlines. Its invalid output and generated corpus are preserved under `results/phase7b-corpus-validation-v2-invalid-r1/` and `data/generated-v2-invalid-r1/`. The evaluator now compares whitespace-normalized text while still rejecting changed content, with a regression test. Validation version `phase7b-corpus-validation-v2-2` uses the **same corpus hash** as invalid r1. No query, gold, retrieval setting, or gate was changed after the first retrieval run. No transition faults were run.

## Phase 7C corpus v3 realism validation

V3 is separate from all v1/v2 artifacts. It has 390 current logical documents, 450 stored versions, 216 questions, eight source archetypes, scoped defaults and independently approved companion records, draft/FAQ distractors, eight version-change types, variable authority, and tenant assignment that does not determine either scope axis. It uses the frozen 80-word chunker, BM25, the pinned local E5 snapshot, top-k 10, and unchanged RRF fusion. The gate thresholds are frozen in `configs/phase7c-enterprise-v3.json`.

The first v3 construction (`phase7c-enterprise-realism-v3-1`) passed its numeric gates but retained the same body layout across archetypes. Its artifacts are preserved under `data/generated-v3-structural-r1/` and `results/phase7c-enterprise-realism-v3-structural-r1/`. The corrected, frozen `phase7c-enterprise-realism-v3-2` generator adds archetype-specific sectioning and prose formats. It has a new corpus hash. **Its clean validation fails R7, R8, and R9; no transition faults were run.** This result is preserved without tuning the corpus or retrievers after observing the failure.

To reproduce v3 into fresh directories from the repository root, with the local E5 snapshot available:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python -m ragstate.campaign.phase7c generate \
  --config configs/phase7c-enterprise-v3.json \
  --data data/generated-v3-reproduction \
  --output results/phase7c-enterprise-realism-v3-reproduction
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python -m ragstate.campaign.phase7c validate \
  --config configs/phase7c-enterprise-v3.json \
  --data data/generated-v3-reproduction \
  --output results/phase7c-enterprise-realism-v3-reproduction
```

The validation reports frozen document-level CER/SCRS and a separate claim-bearing chunk check. With longer documents, a retrieved document can count toward CER while its answer-bearing chunk is absent. The supplementary `scripts/phase7c_evidence_audit.py` derives observed chunk positions from the frozen corpus and records claims split across the chunk boundary. Do not infer context-level evidence availability from document-level CER alone. The failed clean gates prevent a transition campaign on this corpus.

## Phase 7D retrieval architecture comparison

The frozen v3 corpus, queries, gold, metrics, BM25/E5/hybrid families, and RRF settings remain unchanged. The predeclared comparison in `configs/phase7d-architecture-v1.json` evaluates RA0 (80-word chunks), RA1 (160-word chunks), RA2 (deterministic sections bounded at 200 words), and RA3 (top eight documents followed by top ten chunks). The existing chunker has zero overlap, which RA1 preserves. RA3 ranks documents from source text and then ranks chunks from those documents; no oracle or gold identifier enters either ranking stage.

The evidence-span oracle derives each current gold claim's exact sentence and records which candidate chunks contain the complete span. `CBR@10` and all-required claim-bearing success are separate diagnostics; document-level CER and SCRS keep their original definitions. The run compares all 216 questions under all four architectures and three retrievers, repeats rankings, records paired bootstrap confidence intervals and exact McNemar tests, and generates eight SVG/PNG figure pairs. It does not run transitions or an LLM.

To reproduce in a fresh directory with the pinned local E5 snapshot:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python -m ragstate.campaign.phase7d freeze \
  --config configs/phase7d-architecture-v1.json \
  --output results/phase7d-retrieval-architecture-reproduction
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-dense/bin/python -m ragstate.campaign.phase7d run \
  --config configs/phase7d-architecture-v1.json \
  --output results/phase7d-retrieval-architecture-reproduction
```

The result is preserved at `results/phase7d-retrieval-architecture-v1/`. The frozen selection rule chooses RA1, but it fails the multi-document document-level gate A4 (0.833 against 0.85). No architecture passes all seven gates, so the state-transition pilot remains stopped. The selected RA1 result is a diagnostic choice, not approval to launch faults.

## Frozen transition-eligible cohort

`data/transition-cohort-v1/` records a deterministic clean-state inclusion rule for the selected primary architecture: RA1 fixed 160-word chunks, frozen BM25/E5 hybrid, top 10, and no reranker. It reads the preserved Phase 7E RR0 clean output and checks every context against canonical v3 truth and complete claim spans. The Phase 7D RA1 hybrid context and SCRS must match row by row. Neither corpus data nor retrieval ranking is changed by cohort construction.

The primary transition analysis is declared in `analysis_policy.json` to use only the frozen RA1 hybrid and its clean-eligible queries. Negative queries have a separate eligibility rule and ID list. Full-corpus clean analysis continues to include **all** v3 questions and their clean failures. BM25 and E5 are reserved for secondary retriever-specific clean-eligible cohorts if those are frozen separately; they are not in the primary causal factorial. Inclusion uses no faulted-state outcomes or manually selected query IDs.

The cohort contains 183/204 answerable queries and 12/12 negative queries. Its SHA-256 is `2b920c186513a7a13b98d2d5ba4c9b1c61530a61d19f37b65d8c7305eb16a804`. F2 fails because Finance retains 27/34 answerable queries (79.4%) against the frozen 80% minimum. Historical Phase 7D gate A4 remains **FAIL** (25/30 < 0.85); cohort eligibility does not revise it. No transition fault campaign has been run.

To verify the frozen cohort without changing it:

```bash
PYTHONPATH=src .venv-dense/bin/python -m ragstate.campaign.transition_cohort verify \
  --config configs/transition-cohort-v1.json \
  --output data/transition-cohort-v1
```

For a fresh reproduction, use the `generate` stage with a new empty output directory, then compare its cohort SHA-256 to the frozen manifest. The generator refuses to overwrite an existing cohort.

## Phase 7G parent-aware clean comparison

The frozen pre-transition comparison at `results/phase7g-parent-aware-v1/` evaluates exactly four generic policies on the same v3 corpus and frozen RA1 hybrid top-30 candidate pool: PA0 keeps hybrid order; PA1 caps each logical document at two chunks; PA2 fuses chunk and independently retrieved full-document hybrid ranks with the frozen RRF constant; PA3 adds the cap after PA2. Each produces at most ten final chunks. Parent retrieval uses source document text and the same BM25/E5 model pair; no query family, domain, gold label, or claim oracle enters ranking.

The predeclared selection rule chooses PA2. Its clean SCRS is 191/204 versus PA0's 190/204, and its multi-document document success is 26/30 versus 25/30. Multi-document claim success remains 25/30. All G1–G8 clean gates pass for PA2, while the paired confidence interval for the overall SCRS difference includes zero. PA1 produces exactly the same top-ten lists as PA0 across all 216 queries, so the two-chunk cap by itself does not support the crowding hypothesis on this corpus. PA2 and PA3 have identical query-level correctness; PA2 wins the final latency tie-breaker. These are clean retrieval observations, not transition-fault results.

Only after selection, the unchanged cohort-v1 eligibility rules were applied to PA2, producing `data/transition-cohort-v2/`. Its hash is `e74d643f6c2eb7c8ed8c52683e0db0152f7d7ddbcf29d86c070a28e22c79ce43`; 185/204 answerable and 12/12 negative queries are eligible. Finance retains 28/34, so F2 passes for cohort v2. Cohort v1 remains immutable, with F2 **FAIL**. No transition faults or LLM generation were run in Phase 7G.
# Scenario applicability control (pre-science)

`scenario-applicability-v1` compiles the frozen Track-A scenario candidates
against transition, fault, architecture, and replica-structure preconditions.
It reads the frozen v3 corpus and candidate manifest, never the invalid v2/v3
observation rows. It does not run a scientific campaign.

```bash
PYTHONPATH=src .venv-dense/bin/python -m ragstate.campaign.scenario_applicability
```

Replica units are plane-specific: content uses chunks and parent records,
authorization uses per-chunk ACL entries, and C1 uses candidate-cache entries
and their dependencies. Partial delivery uses `floor(N * p)` with no clamping;
the operator is applicable only when `1 <= floor(N * p) < N` and the actual
replica transformation leaves both updated and stale units. An impossible
partial case is excluded with a recorded reason, never converted to delay.

The compiler records every considered candidate, every exclusion, and a
state-only realizability audit in `results/scenario-applicability-v1/`.
AP1–AP5 must all pass before freezing a new scientific campaign. In this
version AP4 fails: the 25% partial condition has only one eligible query per
tested transition, all in Engineering. The executable manifest is therefore
an applicability artifact, not an approved Track-A campaign manifest.

`scenario-applicability-v2` replaces only TA2's universal PARTIAL percentages
with states based on each target's replica units. `P_LOW` updates one unit;
`P_HIGH` updates all but one; `P_MID` uses `round_half_up(N/2)` and is retained
only when distinct. The selected units are SHA-256 ranked from the frozen seed,
query, transition, severity, plane, and unit identity. Neither gold claims nor
evidence position enter that ranking. Every applicable partial case links to
the same query/transition/architecture/severity uniform-delay control.

```bash
PYTHONPATH=src .venv-dense/bin/python -m ragstate.campaign.scenario_applicability_v2
```

The v2 output is under `results/scenario-applicability-v2/`, including the
machine-readable H4 policy and all candidate, exclusion, topology, realized
fraction, matching, compound, and gate tables. Its AP1–AP5 gates validate
applicability and state-only operator realizability. A future scientific
campaign must wire the versioned topology operator into its runtime and audit
the resulting stage traces before collecting scientific outcomes.

## Track-A v4 scientific pilot

The frozen v4 runner consumes the executable scenarios and selected-unit plans
from `scenario-applicability-v2`. Its wiring preflight passed W1–W6 before
science began. The completed campaign has 4,242 scenarios and 63,630
observations under `results/state-isolated-pilot-v4/`; Track B was not run.
The campaign runtime source is locked by the scientific manifest. Existing
results must not be overwritten.

For read-only verification of the frozen campaign and fresh-process subset:

```bash
PYTHONPATH=src .venv-dense/bin/python -m ragstate.campaign.state_pilot_v4_analysis reproduce
PYTHONPATH=src .venv-dense/bin/python -m ragstate.campaign.state_pilot_v4_analysis analyze
```

The analysis command checks campaign-wide hard validity before deriving
tables and figures. The predeclared primary H4 contrast is **not identifiable**
in this run: at the S2 delivery tick, each PARTIAL arm is mixed while its
matched uniform-delay control has fully propagated. The paired table reports
the observed contrast with this timing label; it must not be interpreted as
mixed versus uniformly stale at the same logical time. Revising that contrast
requires a newly frozen pilot design, while v4 artifacts remain preserved.

## H4 matched-horizon supplement

`h4-partial-control-pilot-v1` is a separate Track-A supplement built from the
752 frozen applicability-v2 PARTIAL plans. Each A0 pair holds the query,
transition, target units, and final delivery tick fixed. The uniform arm
delivers all units at T=5; the mixed arm delivers its selected strict subset
at t=2 and the remainder at T=5. The primary comparison window is t=2–4.
The source applicability matrix supplies only S2 PARTIAL cases; S1 and S3
are not represented in this supplement. Track-A v4 remains the scientific
source for all non-H4 analyses and is not modified or rerun here.

The frozen manifest and first failed preflight attempt are retained under
`results/h4-partial-control-pilot-v1/`. The successful preflight passed all
H4-P1–H4-P7 gates, and all 1,504 arms passed full schedule and boundary
audits. The mixed-window binary state-failure endpoint is saturated in both
arms; the tables therefore report that null alongside claim availability and
mixed-version integrity effects. No Track B or LLM generation ran.

Read-only analysis and fresh-process sample commands:

```bash
PYTHONPATH=src .venv-dense/bin/python -m ragstate.campaign.h4_partial_analysis sample_rows
PYTHONPATH=src .venv-dense/bin/python -m ragstate.campaign.h4_partial_analysis analyze
```

