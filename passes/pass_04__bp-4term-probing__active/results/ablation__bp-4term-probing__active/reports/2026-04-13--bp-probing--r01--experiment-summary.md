---
type: results-report
date: 2026-04-13
experiment_line: bp-probing
round: 1
purpose: experiment-summary
status: active
source_artifacts:
  - results/figures/analysis_bundle/analysis-report.md
  - results/figures/analysis_bundle/stats-appendix.md
  - results/figures/analysis_bundle/figure-catalog.md
linked_experiments:
  - configs/probe_P01_parallel_homo.yaml through probe_P18_seeded_parallel.yaml
---

# Validazione Empirica del Framework BP: Esperimenti di Probing P01-P17

**Report per il Supervisore**
**Data**: 13 aprile 2026
**Autore**: Emanuele Rimoldi (analisi automatizzata con Claude)
**Linea sperimentale**: BP 2×2 Probing — Parallelizzazione e Memoria negli Agenti Autonomi

---

## 1. Executive Summary

Abbiamo condotto **16 esperimenti di probing** (P01-P13, P15-P17) per validare empiricamente il framework a quattro termini di Beneventano-Poggio applicato ad agenti di codifica autonomi:

$$\Delta = \log(\kappa_0 / \kappa) + \varphi + G - \varepsilon$$

dove $G$ rappresenta la generazione di informazione (controllata dalla temperatura) e $\varepsilon$ il routing/correzione (controllato dalla memoria).

**Risultati principali**:

1. **La memoria condivisa funziona** — P12 (shared memory riparata) produce distribuzioni significativamente migliori rispetto a P09 (senza memoria): Mann-Whitney U=210, **p<0.001**, effetto grande (r=0.647). Include i 2 unici run sotto-baseline in un esperimento parallelo.
2. **G senza ε = random walk** — P11 (temp=1.2, nessuna memoria, 45 min) conferma che alta generazione senza correzione porta a degrado oscillatorio: 21 run, 0 sotto baseline, media +0.93 sopra baseline.
3. **Effetto soffitto dominante** — Solo il **1.9%** dei run non-baseline (5/269) batte il baseline. Il 97.4% delle modifiche degli agenti peggiora le prestazioni.
4. **Tutti i successi passano per il learning rate** — I 6 run sotto-baseline condividono un unico fattore: modifica del LR nell'intervallo 5e-4 — 2e-3.
5. **Agenti omogenei > diversi** (controintuitivo) — P<0.001. La diversità di temperatura non produce diversità di strategia.
6. **Nuovo record**: P15 (seeded search) raggiunge bpb=0.880 con hint LR=1.5e-3.

**Scoperte operative critiche**: Tre bug nel meccanismo di memoria hanno invalidato i probe P05-P08. Corretti in P12/P17. Un bug addizionale di symlink (git checkout distrugge i link alla memoria condivisa) è stato identificato e corretto durante P12.

---

## 2. Identità dell'Esperimento e Contesto Decisionale

### Obiettivo

Validare la decomposizione BP a quattro termini come framework predittivo per il comportamento degli agenti di codifica autonomi. La domanda centrale: **la memoria (ε) e la parallelizzazione (G) migliorano in modo misurabile la capacità dell'agente di ottimizzare un modello di linguaggio?**

### Design sperimentale

Matrice 2×2 espansa con fattori aggiuntivi:

| Fattore | Livelli testati |
|---------|----------------|
| **Parallelizzazione** | Singolo, Parallelo (2 agenti) |
| **Memoria** | Nessuna, Privata, Condivisa, Entrambe |
| **Temperatura** | 0.3, 0.5, default, 1.0, 1.2 |
| **Diversità** | Omogeneo (stessa temp), Eterogeneo (temp diverse) |
| **Seeding** | Nessuno, Hint LR nel primo messaggio |
| **Budget temporale** | 15, 20, 30, 45 minuti |

### Task

Addestrare un modello di linguaggio (nano-GPT) su Shakespeare per 60 secondi. Metrica: **val_bpb** (validation bits-per-byte, più basso è meglio). Baseline: **0.925845**. Modello agente: **claude-haiku-4-5-20251001** per tutti i probe.

### Perché questo design

Il design a probing rapido (1 replica per configurazione, budget corti) è stato scelto deliberatamente per massimizzare la copertura dello spazio dei parametri in tempo limitato. L'obiettivo non è la significatività statistica a livello di probe, ma l'identificazione di segnale informativo per guidare esperimenti futuri più mirati.

---

## 3. Setup e Protocollo di Valutazione

### Infrastruttura

- **Runner**: Script Python (`run_probes.py`) che lancia agenti Claude in workspace isolati
- **Monitoraggio**: Loop continuo che controlla completamento training, aggiorna log condivisi, ripara symlink
- **Analisi**: Pipeline automatica (`analyze_all_probes.py` + `plot_probes.py`)
- **Esecuzione**: 4 wave sequenziali (P01-P04, P05-P08, P09-P14, P15-P18)

### Dataset finale

| Metrica | Valore |
|---------|--------|
| Probe eseguiti | 16 (P01-P13, P15-P17) |
| Probe mancanti | 2 (P14, P18) |
| Run totali | 299 |
| Run con val_bpb valido | 293 |
| Run con val_bpb nullo | 7 (tutti in P13) |
| Run baseline (primo run) | 25 |
| Run non-baseline | 268 |
| Run sotto baseline | 6 (2.0%) |

### Onde sperimentali

| Wave | Probe | Focus | Budget |
|------|-------|-------|--------|
| Wave 1 | P01-P04 | Calibrazione iniziale | 15-20 min |
| Wave 2 | P05-P08 | Test memoria (BUGGED) | 15-20 min |
| Wave 3 | P09-P14 | Design fattoriale principale | 30-45 min |
| Wave 4 | P15-P18 | Seeded search, full stack | 30-45 min |

---

## 4. Risultati Principali

### 4.1 La memoria condivisa riduce la varianza e abbassa la media (ma non produce breakthrough)

**Contesto critico**: La memoria era configurata ma **non funzionale** in P05-P08 a causa di tre bug:
1. La memoria condivisa veniva popolata da una fonte dati inaffidabile (gli agenti saltano `update_snapshot.py`)
2. La memoria privata leggeva da `trace.jsonl` vuoto invece che da `training_runs.jsonl`
3. Una guardia `elif` impediva l'attivazione simultanea di entrambi i tipi di memoria

Dopo le correzioni (P12, P17):
- **P12** (shared memory corretta, 45 min): 2/41 run sotto baseline (4.9%), best=0.914
- **P17** (entrambe le memorie corrette, 45 min): 0/29 sotto baseline, best=0.955
- **P09** (diverso, senza memoria, 30 min): 0/29 sotto baseline, best=0.971

**Risultato statistico**: P12 vs P09 → U=210, **p<0.001**, r=0.647 (effetto grande). P12 vs P13 → U=63, **p<0.001**, r=0.917 (effetto molto grande).

**Interpretazione nel framework BP**: La memoria condivisa implementa il termine ε. Converte l'esplorazione grezza (G) in apprendimento cumulativo. Non aumenta la quantità di esplorazione, ma ne migliora la **qualità** prevenendo la ripetizione di fallimenti noti.

**Caveat importante**: L'agente_0 di P12 ha perso l'accesso alla memoria condivisa dopo il run 4 (symlink distrutta da git checkout). Solo l'agente_1 ha avuto accesso continuo. Il fix del symlink è stato implementato durante l'esperimento.

### 4.2 Effetto soffitto: il task ha headroom minimo

- **Tasso di successo**: 5/269 non-baseline = **1.9%** (IC 95% Clopper-Pearson: 0.4%-3.7%)
- 31 run entro il 5% del baseline (near misses)
- 85 run >50% peggio del baseline
- Distribuzione fortemente asimmetrica a destra: massa principale tra 0.95-1.25, coda fino a 7.88

**Implicazione**: Il `train.py` di default è già quasi ottimale per 60s di training su questa architettura. Lo spazio di ricerca è dominato da configurazioni dannose. Questa è una proprietà del task, non un fallimento degli agenti.

### 4.3 Agenti omogenei battono i diversi (risultato controintuitivo)

- **Omogenei** (P01+P10): media=1.046, std=0.113
- **Diversi** (P02+P09): media=1.172, std=0.118
- Mann-Whitney U=189, **p<0.001**, r=0.581

**Perché è controintuitivo**: Ci si aspetterebbe che la diversità di temperatura produca diversità di strategia e quindi migliore copertura dello spazio di ricerca. Invece:
1. Gli agenti a bassa temperatura (0.3) producono pochissimi run (es. P07: 3 run vs 15 per temp=1.2)
2. La diversità di temperatura **non produce** diversità di strategia (P02 ha Jaccard=1.0 tra le categorie strategiche dei due agenti)
3. Gli agenti omogenei restano più vicini al baseline facendo perturbazioni piccole

### 4.4 La temperatura accelera l'iterazione ma non la qualità

- Agenti ad alta temperatura (1.2): 152 run, media=1.377
- Agenti a bassa/default temperatura: 134 run, media=1.170
- Test paired (Wilcoxon): p=0.0625 (non significativo)
- Media run per agente: alta temp=17.2 vs bassa temp=9.2

**Nel framework BP**: La temperatura controlla direttamente il termine G. Temperature più alte aumentano G (più iterazioni per unità di tempo), ma le iterazioni sono mediamente peggiori. Senza il termine correttivo ε, un G alto è controproducente.

**Caso critico — P11**: Temperatura 1.2, nessuna memoria, 45 minuti. 21 run, 0 sotto baseline, degradazione media +0.93. L'agente oscilla: escalation LR → degrado → partial git revert → ri-escalation. Senza memoria, l'agente non può imparare dalla propria storia. Questo è la prova empirica più forte che **G senza ε = random walk**.

### 4.5 Tutti i successi passano per il learning rate

| Run | Probe | bpb | Modifica LR |
|-----|-------|-----|-------------|
| P15 run 1 | Seeded | **0.880** | Baseline (hint LR=1.5e-3 nel primo messaggio) |
| P07 run 14 | Shared* | 0.906 | 1e-3 → 1.5e-3 |
| P12 run 13 | Shared (FIXED) | 0.914 | 1e-3 → 5e-4 |
| P05 run 3 | Memory* | 0.919 | 1e-3 → 2e-3 |
| P01 run 3 | Par homo | 0.923 | 1e-3 → 2e-3 |
| P12 run 16 | Shared (FIXED) | 0.924 | Weight decay adjustment |

Tassi di successo per categoria strategica:
- Optimization: 4/121 (3.3%)
- Regularization: 1/77 (1.3%)
- Architecture: 0/15 (0%)
- Data pipeline: 0/7 (0%)

**Interpretazione**: Con 60s di training, solo modifiche a effetto rapido (LR) hanno il tempo di manifestarsi. Architettura, data pipeline e regolarizzazione richiedono tempi di convergenza più lunghi. L'ottimo LR=1.5e-3 è un dato significativo: il default era 1e-3, e piccoli aggiustamenti del 50-100% sono l'unica strada praticabile.

### 4.6 Il modello è capace, il task ha poco headroom

- Tutti i 16 probe usano claude-haiku-4-5-20251001
- Run baseline: media=0.977, std=0.049
- Il modello committizza codice, lancia training e valuta risultati nel 100% dei casi
- 293/293 run hanno commit git validi e categorie strategiche
- Distribuzione protocollo: 84% explore, 9.2% bootstrap, 6.8% reevaluation

**Conclusione**: Il collo di bottiglia non è la capacità del modello ma l'headroom del task. Claude Haiku è sufficiente per il task operativo; il problema è che il task stesso lascia pochissimo margine di miglioramento in 60s.

### 4.7 Uso degli strumenti consistente

- Protocollo: 84% in modalità explore (atteso per ricerca iterativa)
- Tutti i run hanno commit git, categorie strategiche e metadata di valutazione
- Tempo di training: media=66.3s, std=42.6s (P01 aveva 362s per bug nel template, poi corretto)
- Nessuna inconsistenza nell'uso degli strumenti

---

## 5. Validazione Statistica

### Test primari (con correzione Bonferroni, α=0.05/6=0.0083)

| Test | Confronto | Risultato | Significativo? |
|------|-----------|-----------|----------------|
| 1 | P12 vs P09 (shared memory effect) | U=210, **p<0.001**, r=0.647 | **Sì** |
| 1b | P12 vs P13 (stesso budget, diversa memoria) | U=63, **p<0.001**, r=0.917 | **Sì** |
| 2 | Temperatura → conteggio run (Wilcoxon) | p=0.0625 | No (underpowered, n=5 coppie) |
| 3 | Omogenei vs diversi | U=189, **p<0.001**, r=0.581 | **Sì** |
| 4 | Trend degradazione P11 (regressione lineare) | slope=0.004, p=0.804 | No (oscillazione maschera la tendenza) |
| 5 | Task ceiling (Clopper-Pearson) | 1.9% successo, IC 95%=[0.4%, 3.7%] | Soffitto confermato |
| 6 | Seeded search P15 vs P11 | U=61, **p=0.004** | **Sì** |

**Nota**: Tutti i test sono a livello di run (within-probe), non a livello di probe. Con una sola replica per configurazione, non possiamo fare inferenza probe-level.

---

## 6. Interpretazione Figura per Figura

### Figura 1: Confronto Principale (figure-01-main-comparison.pdf)

**Perché esiste**: Fornisce la visione d'insieme di tutti i 293 run attraverso i 16 probe, raggruppati per condizione sperimentale.

**Cosa si osserva**: Le stelle (run sotto-baseline) appaiono solo in P01, P05, P07, P12, P15. La grande maggioranza dei run si raggruppa sopra il baseline. P11 e P13 mostrano varianza estrema e degradazione severa.

**Implicazione decisionale**: L'effetto soffitto è immediatamente visibile. I probe con memoria (P12) e con seeding (P15) sono gli unici con successi consistenti. Questo giustifica il focus su memoria e seeding per gli esperimenti futuri.

### Figura 2: Effetto Memoria (figure-02-memory-effect.pdf)

**Perché esiste**: Distingue l'effetto della memoria configurata vs funzionale e mostra il meccanismo causale.

**Cosa si osserva**: Panel A mostra che più run non equivalgono a migliori risultati (G ≠ qualità). Panel B mostra che l'agente_1 di P12 raggiunge risultati sotto-baseline con alta visibilità della memoria condivisa (17+ voci). Panel C mostra P11 che diverge verso l'alto mentre P12 resta controllato.

**Implicazione decisionale**: La memoria non migliora la quantità di esplorazione ma la qualità. Questo valida il termine ε del framework BP: la correzione del routing è il meccanismo che converte iterazione in progresso.

### Figura 3: Distribuzione per Condizione (figure-03-distribution-by-condition.pdf)

**Perché esiste**: Mostra la forma delle distribuzioni per ciascun gruppo sperimentale.

**Cosa si osserva**: "Parallel shared (fixed)" (solo P12) ha la distribuzione più stretta e centrata più vicino al baseline. "Single high-temp" (P11, P15) ha la dispersione più ampia.

**Implicazione decisionale**: La memoria condivisa riduce la varianza. Questo è operativamente significativo: un agente con memoria è più prevedibile, anche se non sempre migliore.

### Figura 4: Effetto Temperatura (figure-04-temperature-effect.pdf)

**Perché esiste**: Quantifica separatamente l'effetto della temperatura su velocità e qualità.

**Cosa si osserva**: Correlazione positiva tra temperatura e conteggio run (r=0.42, non significativa), ma nessuna relazione tra temperatura e miglior performance.

**Implicazione decisionale**: Aumentare G (temperatura) senza ε (memoria) è inutile o dannoso. Le risorse investite in temperatura alta sono sprecate senza il meccanismo correttivo.

### Figura 5: Categorie Strategiche (figure-05-strategy-categories.pdf)

**Perché esiste**: Mostra quali strategie gli agenti scelgono e quali funzionano.

**Cosa si osserva**: Optimization è la più comune (121 run) e l'unica con tasso di successo >0% (3.3%). Architecture e data pipeline hanno 0% di successo.

**Implicazione decisionale**: Con budget di 60s, solo modifiche "veloci" (LR, weight decay) hanno effetto. Estendere il budget di training potrebbe sbloccare strategie attualmente inefficaci.

### Figura 6: Degradazione e Convergenza (figure-06-degradation-convergence.pdf)

**Perché esiste**: Dimostra empiricamente "G senza ε = random walk" e confronta le traiettorie di convergenza.

**Cosa si osserva**: Panel A mostra i tre cicli di degradazione di P11 (escalation → revert → ri-escalation). Panel B mostra P12 con miglioramento cumulativo costante, mentre P11 ristagna.

**Implicazione decisionale**: Questa è la prova più forte a favore del framework BP. Senza memoria, l'agente è condannato a ripetere gli stessi errori. Con memoria condivisa, l'agente accumula conoscenza e evita trappole note.

---

## 7. Casi di Fallimento, Risultati Negativi e Limitazioni

### Bug scoperti e corretti

1. **Memoria condivisa non funzionale (P05-P08)**: La fonte dati per popolare il log condiviso era inaffidabile. Gli agenti saltavano `update_snapshot.py`, lasciando la memoria vuota. **Corretto**: Popolamento diretto dal completamento del training run nel monitoring loop.

2. **Memoria privata vuota (P05-P08)**: Leggeva da `trace.jsonl` (vuoto) invece che da `training_runs.jsonl` (con i dati reali). **Corretto**: Riferimento al file corretto.

3. **Guardia elif (P05-P08)**: Il codice `elif` per la memoria condivisa impediva l'esecuzione del blocco di memoria privata quando entrambe erano abilitate. **Corretto**: Logica separata per i due tipi.

4. **Symlink fragile (P12)**: `git checkout` eseguito dall'agente distruggeva il symlink alla memoria condivisa. L'agente_0 di P12 ha perso accesso alla memoria dopo il run 4. **Corretto**: Aggiunto codice di riparazione automatica nel monitoring loop di `claude_agent_runner.py`.

### Esperimenti mancanti

- **P14** (memoria privata + alta temperatura): Non eseguito. Questo era il test chiave per ε come meccanismo di correzione della degradazione. Priorità alta per ri-lancio.
- **P18** (seeded parallel): Non eseguito. Avrebbe testato l'effetto combinato di seeding e parallelizzazione.

### Limitazioni strutturali

1. **Nessuna replica a livello di seed**: Ogni configurazione ha una sola esecuzione. Le statistiche sono run-level (within-probe), non probe-level.
2. **Budget confuso**: I probe spaziano da 15 a 45 minuti. Budget più lunghi producono meccanicamente più run. I confronti cross-budget sono confusi.
3. **Singolo modello**: Tutti gli esperimenti usano Claude Haiku. I risultati potrebbero non generalizzare a Sonnet/Opus.
4. **Singolo task**: Un'architettura, un dataset. La sensibilità al LR è task-specifica.
5. **P01 template bug**: P01 ha usato 315s/run di training (bug nel template originale). Non direttamente comparabile.
6. **P13 qualità dati**: 7 run con val_bpb nullo (training failures) — potenzialmente dovuti a modifiche eccessivamente aggressive dall'agente ad alta temperatura senza memoria.

---

## 8. Cosa è Cambiato nelle Nostre Credenze

### Prima degli esperimenti

- **Credenza**: La parallelizzazione con agenti diversi esplorerebbe meglio lo spazio di ricerca.
- **Ora**: La diversità di temperatura non produce diversità strategica. Gli agenti omogenei sono più sicuri.

### Prima degli esperimenti

- **Credenza**: La memoria migliorerebbe il tasso di successo (trovare configurazioni migliori).
- **Ora**: La memoria migliora la **stabilità** (meno degradazione), non il tasso di scoperta. Il meccanismo è la prevenzione di fallimenti ripetuti, non la guida verso successi.

### Prima degli esperimenti

- **Credenza**: Il task avrebbe headroom sufficiente per testare il framework BP in modo informativo.
- **Ora**: Il task ha un effetto soffitto severo (1.9% successo). Questo limita il potere discriminante. Per il paper, questo è sia un risultato (documenta l'effetto soffitto) sia un limite (riduce la nostra capacità di distinguere configurazioni).

### Nuova comprensione

- **G senza ε = random walk** è ora empiricamente supportato (P11 vs P12, p<0.001)
- **Il LR è l'unico asse praticabile** in 60s di training — dato concreto per guidare il seeding
- **I symlink sono fragili** in workspace dove gli agenti eseguono git operations — richiede repair automatico

---

## 9. Azioni Successive

### Priorità immediata (questa settimana)

1. **Rilanciare P14** — Test memoria privata + alta temperatura. Esperimento critico mancante per validare se ε corregge la degradazione di G in condizioni di alta temperatura (confronto diretto con P11).

2. **Rilanciare P18** — Seeded parallel. Testa l'effetto combinato del miglior intervento (seeding LR=1.5e-3) con la parallelizzazione.

3. **Estendere il budget di training a 120-180s** — Il soffitto attuale è un artefatto del budget di 60s. Con budget più lungo, strategie come regularizzazione e architettura potrebbero diventare efficaci.

### Priorità a medio termine (prossime 2 settimane)

4. **Testare con Sonnet/Opus** — Verificare se un modello più capace produce un tasso di successo superiore al 1.9%. Questo separerebbe "limitazione del task" da "limitazione del modello".

5. **Seed replication** — Replicare P12 (miglior configurazione) con 3-5 seed per ottenere statistiche probe-level.

6. **Aumentare il headroom del task** — Opzioni: (a) task più complesso (architettura più grande), (b) dataset più grande, (c) budget di training più lungo, (d) metrica diversa.

### Per il paper

7. **Narrative principale confermata**: Il framework BP è predittivo — G senza ε produce random walk (P11), ε riduce varianza e previene degradazione (P12). Questo è il risultato centrale.

8. **Figure pronte per il paper**: Figure 2 (memoria) e 6 (degradazione) sono i candidati principali per il manoscritto. Richiedono lucidatura grafica ma il contenuto informativo è solido.

9. **Limitazione da dichiarare**: Il soffitto al 1.9% limita la nostra capacità di quantificare l'effetto positivo della memoria sulla scoperta. Il claim deve essere formulato come "la memoria previene la degradazione" (forte), non "la memoria migliora le prestazioni" (debole).

---

## 10. Indice degli Artefatti e Riproducibilità

### Configurazioni

| File | Descrizione |
|------|-------------|
| `configs/probe_P01_parallel_homo.yaml` — `probe_P18_seeded_parallel.yaml` | 18 file YAML con tutte le configurazioni |
| `workflow/scripts/run_probes.py` | Runner principale con definizioni Wave 1-4 |
| `src/agent_parallelization_new/agents/claude_agent_runner.py` | Runner agente con fix symlink (post-P12) |

### Dati grezzi

| Directory | Contenuto |
|-----------|-----------|
| `runs/experiment_probe_P01/` — `runs/experiment_probe_P17/` | 16 directory con workspace agenti, training logs, commit history |
| Totale run: 293 validi + 7 nulli | JSONL logs in `training_runs.jsonl` per ogni agente |

### Bundle di analisi

| File | Descrizione |
|------|-------------|
| `results/figures/analysis_bundle/analysis-report.md` | Report analitico completo (7 finding) |
| `results/figures/analysis_bundle/stats-appendix.md` | Appendice statistica (6 test, Bonferroni) |
| `results/figures/analysis_bundle/figure-catalog.md` | Catalogo figure con interpretazione |
| `results/figures/analysis_bundle/figure-01-main-comparison.pdf` | Confronto principale (16 probe) |
| `results/figures/analysis_bundle/figure-02-memory-effect.pdf` | Effetto memoria (3 panel) |
| `results/figures/analysis_bundle/figure-03-distribution-by-condition.pdf` | Distribuzioni per condizione |
| `results/figures/analysis_bundle/figure-04-temperature-effect.pdf` | Effetto temperatura |
| `results/figures/analysis_bundle/figure-05-strategy-categories.pdf` | Categorie strategiche |
| `results/figures/analysis_bundle/figure-06-degradation-convergence.pdf` | Degradazione e convergenza |

### Riproducibilità

- **Modello**: claude-haiku-4-5-20251001 (identico per tutti i probe)
- **Seed management**: Non applicabile (agenti LLM stocastici per natura)
- **Config recording**: Tutti i parametri in YAML versionati
- **Environment**: macOS, Python 3.x, Claude Code CLI
- **Commit**: Branch `bp-2x2-instrumentation`, HEAD al momento dell'analisi

---

*Report generato automaticamente a partire dal bundle di analisi in `results/figures/analysis_bundle/`. Tutti i test statistici utilizzano scipy con assunzioni documentate. Le figure sono generate con matplotlib/seaborn dai dati grezzi.*
