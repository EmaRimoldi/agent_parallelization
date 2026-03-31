# Original Repo Reverse Engineering

Repo path: `/home/erimoldi/projects/agent_parallelisation`
Results root: `/home/erimoldi/projects/agent_parallelisation/results`
Trajectory files found: 22

## Best result
- Run: `exp_smart_20260330_063836` / Agent: `agent_0`
- val_bpb: `1.1020746984708296`

## Known best (from build guide)
- val_bpb: `1.1020746984708296`
- Run: `exp_smart_20260330_063836` / Agent: `agent_0`
- Snapshot: `iter0003_s350_bpb1.1021.py`

## Hyperparameter changes
- `EMBEDDING_LR` = `0.8`
- `UNEMBEDDING_LR` = `0.005`
- `MATRIX_LR` = `0.06`
- `WEIGHT_DECAY` = `0.1`
- `WARMDOWN_RATIO` = `0.4`

## All trajectories

### exp_prompt_20260329_165455/agent_1
- Entries: 7, Best val_bpb: 1.1205303498680368
  - step=289 val_bpb=1.121353
  - step=289 val_bpb=1.120530
  - step=289 val_bpb=1.125206
  - step=289 val_bpb=1.127391
  - step=289 val_bpb=1.126736
  - ...(2 more)

### exp_prompt_20260329_165455/merge
- Entries: 1, Best val_bpb: 1.122704732079809
  - step=289 val_bpb=1.122705

### exp_prompt_20260329_165455/serial_agent
- Entries: 9, Best val_bpb: 1.1224133547240387
  - step=290 val_bpb=1.122413
  - step=289 val_bpb=1.123009
  - step=289 val_bpb=1.122904
  - step=288 val_bpb=1.122841
  - step=289 val_bpb=1.122465
  - ...(4 more)

### exp_prompt_20260329_221655/agent_1
- Entries: 9, Best val_bpb: 1.1223501323452159
  - step=287 val_bpb=1.123411
  - step=287 val_bpb=1.125006
  - step=287 val_bpb=1.125013
  - step=287 val_bpb=1.126362
  - step=287 val_bpb=1.133665
  - ...(4 more)

### exp_prompt_20260329_221655/merge
- Entries: 1, Best val_bpb: 1.123955630824605
  - step=286 val_bpb=1.123956

### exp_prompt_20260329_221655/serial_agent
- Entries: 18, Best val_bpb: 1.123693108300334
  - step=286 val_bpb=1.124917
  - step=286 val_bpb=1.123750
  - step=287 val_bpb=1.124855
  - step=287 val_bpb=1.123693
  - step=287 val_bpb=1.124566
  - ...(13 more)

### exp_prompt_20260330_125439/merge
- Entries: 1, Best val_bpb: 1.122021480659964
  - step=291 val_bpb=1.122021

### exp_prompt_20260330_125439/serial_agent
- Entries: 1, Best val_bpb: 1.1218803539634197
  - step=291 val_bpb=1.121880

### exp_smart_20260329_221655/agent_1
- Entries: 6, Best val_bpb: 1.1230282762628698
  - step=287 val_bpb=1.123408
  - step=287 val_bpb=1.124951
  - step=287 val_bpb=1.125217
  - step=287 val_bpb=1.126496
  - step=287 val_bpb=1.133614
  - ...(1 more)

### exp_smart_20260330_063836/agent_0
- Entries: 4, Best val_bpb: 1.1020746984708296
  - step=350 val_bpb=1.112890
  - step=350 val_bpb=1.106942
  - step=350 val_bpb=1.106135
  - step=350 val_bpb=1.102075

### exp_smart_20260330_063836/agent_1
- Entries: 2, Best val_bpb: 1.114567399099616
  - step=350 val_bpb=1.117118
  - step=350 val_bpb=1.114567

### exp_smart_20260330_063836/serial_agent
- Entries: 6, Best val_bpb: 1.1207495150660547
  - step=214 val_bpb=1.146238
  - step=144 val_bpb=1.168940
  - step=672 val_bpb=1.120750
  - step=139 val_bpb=1.518398
  - step=115 val_bpb=1.598291
  - ...(1 more)

### exp_temp_20260329_165455/agent_1
- Entries: 8, Best val_bpb: 1.1225098434914553
  - step=289 val_bpb=1.122510
  - step=289 val_bpb=1.122644
  - step=289 val_bpb=1.129884
  - step=288 val_bpb=1.125658
  - step=289 val_bpb=1.125633
  - ...(3 more)

### exp_temp_20260329_165455/merge
- Entries: 1, Best val_bpb: 1.1229477063742592
  - step=289 val_bpb=1.122948

### exp_temp_20260329_165455/serial_agent
- Entries: 15, Best val_bpb: 1.1272494377951925
  - step=287 val_bpb=1.128166
  - step=289 val_bpb=1.127350
  - step=289 val_bpb=1.127573
  - step=290 val_bpb=1.127634
  - step=214 val_bpb=1.147749
  - ...(10 more)

### exp_temp_20260329_221655/agent_1
- Entries: 9, Best val_bpb: 1.1223037505597837
  - step=287 val_bpb=1.123489
  - step=287 val_bpb=1.124703
  - step=287 val_bpb=1.125228
  - step=287 val_bpb=1.126092
  - step=287 val_bpb=1.133549
  - ...(4 more)

### exp_temp_20260329_221655/merge
- Entries: 1, Best val_bpb: 1.1237153844006436
  - step=286 val_bpb=1.123715

### exp_temp_20260329_221655/serial_agent
- Entries: 18, Best val_bpb: 1.123334705095333
  - step=287 val_bpb=1.123335
  - step=286 val_bpb=1.123580
  - step=287 val_bpb=1.123580
  - step=287 val_bpb=1.130847
  - step=287 val_bpb=1.131210
  - ...(13 more)

### exp_temp_20260330_125439/serial_agent
- Entries: 7, Best val_bpb: 1.1224118776378127
  - step=290 val_bpb=1.122412
  - step=178 val_bpb=1.162382
  - step=290 val_bpb=1.125579
  - step=291 val_bpb=1.129319
  - step=291 val_bpb=1.130904
  - ...(2 more)

### prod_20260329_130122/agent_0
- Entries: 8, Best val_bpb: 1.128398400018723
  - step=351 val_bpb=1.138717
  - step=352 val_bpb=1.143841
  - step=351 val_bpb=1.149595
  - step=351 val_bpb=1.148449
  - step=352 val_bpb=1.157551
  - ...(3 more)

### prod_20260329_130122/agent_1
- Entries: 8, Best val_bpb: 1.1282147061891337
  - step=351 val_bpb=1.128215
  - step=351 val_bpb=1.145688
  - step=350 val_bpb=1.155808
  - step=350 val_bpb=1.157123
  - step=350 val_bpb=1.147073
  - ...(3 more)

### prod_20260329_130122/serial_agent
- Entries: 17, Best val_bpb: 1.1064804707550489
  - step=351 val_bpb=1.138515
  - step=351 val_bpb=1.111210
  - step=351 val_bpb=1.121121
  - step=351 val_bpb=1.106480
  - step=351 val_bpb=1.128756
  - ...(12 more)

