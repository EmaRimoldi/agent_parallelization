# Parallel Agent Workflow — Architecture Diagrams

---

## 1. System Architecture (N agents)

```mermaid
flowchart TD
    CFG[/"experiment.yaml\nn_agents · model · budget\ntemperature · SLURM params"/]

    subgraph ORCH["ORCHESTRATOR — orchestrator.py — main Python process"]
        direction TB
        O1["① Read ExperimentConfig from experiment.yaml"]
        O2["② Create N git worktrees\none isolated branch per agent"]
        O3["③ Spawn N IsolatedAgentProcess simultaneously\nno stagger · no stagger · no cross-init"]
        O4["④ Poll is_alive() every 10 s\n⚠ reads NO results\n⚠ shares NOTHING between agents\n⚠ only enforces hard deadline → SIGTERM"]
        O5["⑤ All done → Collector\naggregate results.tsv · trajectory.jsonl\nsnapshots · reasoning traces"]
        O6["⑥ Merge phase\nbest-params train.py → single final SLURM job"]
        O1 --> O2 --> O3 --> O4 --> O5 --> O6
    end

    CFG --> O1

    O3 -- "spawn\n(daemon=True)" --> P0
    O3 -- "spawn" --> P1
    O3 -- "spawn" --> PN
    O4 -. "is_alive()?" .-> P0
    O4 -. "is_alive()?" .-> P1
    O4 -. "is_alive()?" .-> PN

    subgraph P0["Process 0 — multiprocessing.Process"]
        CAR0["ClaudeAgentRunner · agent_0\n(budget clock · backoff · session-id)"]
        L0["Turn loop\n─────────────────────\nTurn 1 : first_message\n         workspace · budget · IDs\nTurn 2+ : 'Continue. ~M min left.\n          Keep experimenting.'\n─────────────────────\nstop when budget.should_stop()"]
        DONE0["write metadata.json"]
        CAR0 --> L0 --> DONE0
    end

    subgraph P1["Process 1 — multiprocessing.Process"]
        CAR1["ClaudeAgentRunner · agent_1"]
        L1["Turn loop\n(same structure)"]
        CAR1 --> L1
    end

    subgraph PN["Process N — multiprocessing.Process"]
        CARN["ClaudeAgentRunner · agent_N"]
        LN["Turn loop\n(same structure)"]
        CARN --> LN
    end

    L0 -- "claude --print\n--system-prompt experiment-runner.md\n--dangerously-skip-permissions\n'[turn message]'" --> SA0
    SA0 -- "stdout text\n(logged to run_agent.log)" --> L0

    subgraph SA0["Sub-agent 0 — claude --print subprocess"]
        direction TB
        I1["① read train.py"]
        I2["② form hypothesis\none scalar param · one new value"]
        I3["③ edit train.py"]
        I4["④ git commit"]
        I5["⑤ python save_snapshot.py\n→ snapshots/step_N/train.py\n→ reasoning/trace.jsonl (open entry)"]
        I6["⑥ bash submit_training.sh\n→ SLURM job_id"]
        I7["⑦ poll check_training.sh every 30 s\nuntil TRAINING DONE or FAILED"]
        I8{"val_bpb\nimproved?"}
        I9K["keep commit\nstatus = keep"]
        I9R["git reset --hard HEAD~1\nstatus = revert"]
        I10["⑧ python update_snapshot.py\n→ accepted · reason · next_step\n→ reasoning/trace.jsonl (close entry)"]
        I11["⑨ append to\nresults.tsv · trajectory.jsonl"]
        I1 --> I2 --> I3 --> I4 --> I5 --> I6 --> I7 --> I8
        I8 -- yes --> I9K
        I8 -- no  --> I9R
        I9K --> I10
        I9R --> I10
        I10 --> I11
        I11 -- "loop until\ntime budget" --> I1
    end

    SA0 -- "sbatch" --> SLURM[("SLURM\n(pi_tpoggio)\ngpu:1 per job")]
    SLURM -- "writes train_JOB.out\nval_bpb · peak_vram" --> WS0

    subgraph WS0["workspace_0/ — git worktree — branch: claude/exp/agent_0"]
        direction LR
        F1["train.py\n(modified in place)"]
        F2["results/\nresults.tsv\ntrajectory.jsonl\nmetadata.json"]
        F3["snapshots/\nstep_001/ step_002/ …"]
        F4["reasoning/\ntrace.jsonl"]
        F5["logs/\nrun_agent.log\ntrain_JOB_ID.out"]
    end

    SA0 --> WS0

    WALL["❌  NO CROSS-AGENT COMMUNICATION\n────────────────────────────\nAgents share: nothing\nSeparate git branches · separate processes\nSeparate Claude sessions · separate SLURM jobs\nOrchestrator never forwards any agent result\nto any other agent at any point during the run"]

    style WALL fill:#fff0f0,stroke:#cc0000,color:#880000
    style ORCH fill:#eef2ff,stroke:#4466cc
    style P0   fill:#f0fff4,stroke:#33aa55
    style P1   fill:#f0fff4,stroke:#33aa55
    style PN   fill:#f0fff4,stroke:#33aa55
    style SA0  fill:#fffbea,stroke:#cc9900
    style WS0  fill:#f8f8f8,stroke:#888888
    style SLURM fill:#f0f0ff,stroke:#6666cc
```

---

## 2. Communication Map — what talks to what, and when

```mermaid
sequenceDiagram
    actor User
    participant ORC  as Orchestrator<br/>(main process)
    participant PROCi as Process i<br/>(IsolatedAgentProcess)
    participant CAR  as ClaudeAgentRunner
    participant CLP  as claude --print<br/>(sub-agent)
    participant SLURM as SLURM

    Note over ORC,SLURM: ── SETUP ──────────────────────────────────────────────
    User ->> ORC: python launcher.py --config experiment.yaml
    ORC ->> ORC: create N git worktrees + workspace scripts
    ORC ->> ORC: build first_message from template for each agent

    Note over ORC,SLURM: ── LAUNCH (T = 0, all simultaneous) ────────────────────
    ORC ->> PROCi: Process.start()  ×N  [no stagger, no ordering]
    activate PROCi

    Note over ORC,SLURM: ── RUN PHASE ────────────────────────────────────────────
    loop every 10 s
        ORC -->> PROCi: is_alive()?   [read-only poll, no data exchange]
    end

    PROCi ->> CAR: run(system_prompt, first_message)
    activate CAR

    loop until budget.should_stop()
        CAR ->> CLP: claude --print "[turn message]"<br/>  Turn 1 → first_message (workspace, IDs, budget)<br/>  Turn 2+ → "Continue. ~M min remaining."
        activate CLP

        loop inner research loop (inside one claude --print turn)
            CLP ->> CLP: read train.py
            CLP ->> CLP: edit train.py  +  git commit
            CLP ->> CLP: python save_snapshot.py
            CLP ->> SLURM: bash submit_training.sh  (sbatch)
            SLURM -->> CLP: job_id
            loop poll every 30 s
                CLP ->> SLURM: bash check_training.sh job_id
                SLURM -->> CLP: TRAINING RUNNING / DONE / FAILED
            end
            CLP ->> CLP: compare val_bpb → keep or git reset
            CLP ->> CLP: python update_snapshot.py
            CLP ->> CLP: append results.tsv · trajectory.jsonl
        end

        CLP -->> CAR: stdout text  (logged to run_agent.log)
        deactivate CLP
        CAR ->> CAR: check output for errors / rate-limits<br/>update budget clock
    end

    CAR ->> CAR: write metadata.json
    deactivate CAR
    deactivate PROCi

    Note over ORC,SLURM: ── COLLECTION (after all N processes finish) ─────────────
    ORC ->> ORC: Collector: read all results.tsv · trajectory.jsonl<br/>snapshots · reasoning traces → aggregate reports

    Note over ORC,SLURM: ── MERGE ────────────────────────────────────────────────
    ORC ->> ORC: run_best_params_merge.py:<br/>find best val_bpb per param across all agents
    ORC ->> SLURM: submit merged train.py
    SLURM -->> ORC: val_bpb of merged run

    Note over ORC,SLURM: ── NO AGENT↔AGENT COMMUNICATION AT ANY STEP ─────────────
```

---

## 3. What the orchestrator controls vs. what it never touches

| Orchestrator **does** | Orchestrator **never does** |
|---|---|
| Creates isolated workspaces (git worktrees) | Reads any agent's `results.tsv` during the run |
| Spawns all processes at the same instant | Passes one agent's val_bpb to another |
| Polls `is_alive()` every 10 s | Modifies any agent's `train.py` |
| Sends SIGTERM at hard deadline (3× budget) | Steers any agent's hypothesis choices |
| Calls Collector **after** all agents finish | Communicates with the claude subprocesses directly |
| Runs the merge phase | Shares reasoning traces between agents during the run |

---

## 4. Isolation guarantees

```
agent_0/workspace/   ←── git branch: claude/EXP/agent_0   ──→  separate git history
agent_1/workspace/   ←── git branch: claude/EXP/agent_1   ──→  separate git history
...
agent_N/workspace/   ←── git branch: claude/EXP/agent_N   ──→  separate git history
         ↑                         ↑                                    ↑
   separate OS process       separate claude --print         separate CUDA_VISIBLE_DEVICES
   (multiprocessing)         subprocess per turn             (no GPU contention)
```

All workspaces share **read-only** symlinks to `data/` and `.venv/` (never written by agents).
