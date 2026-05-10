# CLAUDE.md — Project hybrid policy: OMC + Superpowers

This file overrides default skill behavior for this repository.

## Plugin scope (this project)

- **OMC** (`oh-my-claudecode`): runtime multi-agent orchestration, state tracking, autopilot loops
- **Superpowers**: software-engineering methodology (TDD, plan-spec, verification, code review)
- Both are usable; this file resolves overlap and stops double-fires.

## Override 1 — disable `using-superpowers` auto-trigger in autonomous modes

`superpowers:using-superpowers` says "invoke a skill before ANY response, even at 1 % match". This conflicts with OMC's `autopilot` / `ralph` / `ultrawork` continuous execution.

Rule:
- When **inside an OMC autonomous mode** (i.e. `.omc/state/sessions/*/autopilot-state.json` or `ralph-state.json` or `ultrawork-state.json` is `active=true`): **do NOT auto-invoke `using-superpowers`**. The OMC workflow is authoritative.
- When **OMC subagent** (executor / verifier / debugger / …): the skill's own `<SUBAGENT-STOP>` exempts you — already automatic.
- When **manually driven** (no OMC mode active): may consult superpowers normally.

## Override 2 — single source of truth per phase (no double-fire)

Pick **one** path per task; never both:

| Phase | Default | Reason |
|---|---|---|
| Open-ended ideation | `superpowers:brainstorming` | superpowers stronger at intent exploration |
| Ambiguity → spec | `omc:deep-interview` | mathematical ambiguity gate |
| Plan writing | `superpowers:writing-plans` (solo) **OR** `omc:ralplan` (consensus) | not both |
| Plan execution | `omc:autopilot` Phase 2 (autonomous) **OR** `superpowers:executing-plans` (manual checkpoints) | not both |
| Parallel tasks | `omc:ultrawork` | only OMC has state tracking |
| Subagent dispatch | `omc:team` / Task tool | only OMC has team protocol |
| Worktree isolation | `omc:project-session-manager` | OMC version covers superpowers' use case |
| TDD | `superpowers:test-driven-development` (methodology) + `omc:test-engineer` agent (impl) | layered |
| Bug investigation | `omc:trace` (causal evidence) → `superpowers:systematic-debugging` (don't-jump-to-fix reminder) | sequential |
| Test-fix loop | `omc:ultraqa` | runs the actual loop |
| Verification before claim | `superpowers:verification-before-completion` | no OMC equivalent (OMC has agents, not a self-checklist) |
| Code review | `omc:code-reviewer` agent (does review) + `superpowers:requesting-code-review` (asks for it) + `superpowers:receiving-code-review` (consumes feedback) | layered |
| Branch finishing | `superpowers:finishing-a-development-branch` | OMC has no equivalent |

## Override 3 — anti-double-verify

When `omc:ultraqa` is running, **do NOT additionally** invoke `superpowers:verification-before-completion` per turn — ultraqa IS the verification loop. Apply `verification-before-completion` only to **manual** completion claims (e.g. before announcing "done" outside an OMC mode).

## Override 4 — debugging stack

For any unexpected behavior:
1. Start with `superpowers:systematic-debugging` (philosophy: don't fix until you understand)
2. Use `omc:trace` (or the `tracer` agent) to gather evidence with competing hypotheses
3. Only after root cause confirmed → propose fix
4. Verify with `omc:ultraqa` if test loop is needed

## Override 5 — no skill repetition for the same call site

If a skill was already invoked earlier in this session for the same call site (same file, same scope), **do NOT** re-invoke it on subsequent turns unless the situation changed. Especially:
- `using-superpowers` once per session, not per turn
- `brainstorming` once per feature, not per sub-task

## Project-specific facts (preserve)

- Repository: `dentar142/scg-snn-eg4s20` (multi-modal SCG SNN on Anlogic EG4S20)
- Current deployed bit: `build_snn/scg_top_snn_aligned_h32t16.bit` (95.02 % board acc)
- Cross-dataset bit: `build_snn/scg_top_snn_dropout_aligned.bit` (CEBSDB 78 % 0-shot, WESAD 68 %)
- 3 dataset corpora available: `data_foster_multi/all.npz`, `data_excl100/val.npz` (CEBSDB), `data_wesad/all.npz`
- Pareto-optimal config: H=32, T=16
- Modality dropout `p_drop=0.5` is the cross-dataset robustness recipe (verified on 2 datasets)
- All FP32 ckpts (5.3 MB total) committed to repo
- 17 figures + 11 manifests + full bench JSONs in `doc/`
- Toilet badge generator: `dentar142/StoneBadge` (already integrated via GitHub Action)
