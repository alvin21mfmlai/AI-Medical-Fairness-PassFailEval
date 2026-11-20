# Provenance: where each part of passfail-eval comes from

Extracted on 20 Nov 2026 from the EquityMedQA-SG v2 project (`src/emqsg/`, `prompts/`). `sha256` gives the first 16
hex digits of each source file at extraction. Every source file below except two was identical to the copy frozen with
the study's batch 17 (the judge grid of 2–3 Nov 2026), whose results the package was checked against. The two
exceptions, `sources.py` and `consequences.py`, were written after batch 17 for the post hoc analyses of batch 20
(7 Nov 2026), whose results were frozen; the package reproduces those results too.

| Package file | Taken from (project path, function) | sha256 of the source | What changed |
|---|---|---|---|
| `prompts/judge_equitymedqa_sg.txt` | `prompts/phase0_judge_ref.txt` | `f7e77443541a8bb4` | nothing (byte-identical) |
| `prompts/draft_key_equitymedqa_sg.txt` | `prompts/phase0_draft_core.txt` | `439c53c9d467d73a` | its two grounded-mode placeholders are left empty, which is what the study sent in ungrounded mode |
| `prompts/judge_generic.txt`, `draft_key_generic.txt` | new | | domain-neutral versions of the two prompts, with the same parts and reply format, without urgency and missing information |
| `jsonparse.py` | `src/emqsg/jsonparse.py` | `54834f91e804cf9b` | nothing (byte-identical) |
| `rules.py` | `src/emqsg/phase0/judge_ref.py`, `measures` | `71c880e9c639f42a` | `pass` added (= `accuracy_lenient`, or `accuracy` with rule strict); urgency and missing information computed only when the key has them; the urgency scale is a setting |
| `judge.py` | `src/emqsg/phase0/judge_ref.py`, `check_grade`, `judge` | `71c880e9c639f42a` | generic answer and key files instead of the study's inputs; urgency and missing information checked only when the key and the prompt have them; one log per judge and repeat |
| `keys.py` | `src/emqsg/phase0/schema.py`, `with_ids`, `core_as_text` | `55c040e85c26bf45` | `sg_facts` renamed `facts`; optional sections; a loader with checks for hand-written keys |
| `quotes.py` | `schema.py`, `quote_found`; `src/emqsg/judge.py`, `strip_thought` | `55c040e85c26bf45`, `17c0f6a5e963c41d` | `<think>` blocks are removed as well as MedGemma's |
| `draft.py` | `schema.py`, `check_core` (ungrounded); `src/emqsg/phase0/draft.py`, Stage 1 only | `55c040e85c26bf45`, `f08da10836d595ab` | a generic preset; grounded drafting (retrieved guideline passages) and Stage 2 (per-attribute permitted differences) are not included |
| `llm.py` | `src/emqsg/llm.py` | `1f1de45231df6820` | only the OpenAI-compatible and mock backends; the runner, retries, validator feedback and resumable logs are unchanged |
| `io.py` | `src/emqsg/io.py` | `bb8077684e91e437` | without pandas |
| `flips.py` | `src/emqsg/phase0/analyze.py` (`_boot`, `_differ`, `refdiff`), `independent.py` (`_swap_null`), `sources.py` (`lenient_flip`, `base_mean`), `consequences.py` (`q0_key_vs_patient`) | `00922d60f764cd14`, `f34b8ebe9d4aaf03`, `c1510e53803c71b9`, `ec5a244d0ff91597` | the study's separate analyses joined into one ladder with generic pairing (`variant_set`, `twin_of` instead of the study's id conventions); regrade and judge rungs and the per-question table added |
| `mock.py`, `config.py`, `cli.py`, `examples/`, `tests/` | new | | |

## Not included, and why

| Left out | Why |
|---|---|
| Server launch scripts, Docker images, job scripts, GPU telemetry, quantised model variants | specific to the ThinkStation PGX (GB10, aarch64) the study ran on; any OpenAI-compatible server replaces them |
| The study's settings (model ports, reasoning switches per model family) | replaced by one model entry per judge in the config |
| Grounded drafting and its guideline corpus | the corpus is internal to the project |
| The study's data (questions, answers, keys, grades) | not needed to use the method; the example data are synthetic |
| Specialty labels, the doctors' annotation site, key alignment, justified-difference judging | other parts of the study, not the pass/fail approach |

## Checked against the study

The project keeps the check (`share/checks/check_against_study.py` in EquityMedQA-SG v2) and its report. It rebuilt
every judge and drafting prompt from the frozen inputs, re-validated every frozen reply, recomputed every measure and
reproduced 60 published flip statistics, with no difference (see README, Fidelity). A self-test changed one thing at a
time (the key labels, the rule, the number of resamples, the scale, the draft check) and the check failed each time.
