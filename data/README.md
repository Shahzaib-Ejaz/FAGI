# Data

## CICIDS2017

Download `MachineLearningCSV.zip` from the Canadian Institute for Cybersecurity
(https://www.unb.ca/cic/datasets/ids-2017.html) and extract the eight CSV files into `data/cicids2017/`:

```
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

The archive used for the paper has SHA-256 `c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928`
(MD5 `4f83860afbf29cac8163854095bf6cf7`). Subdirectories are fine; the loader searches recursively.

`data/cicids2017.py` reads the files with `skipinitialspace=True` and latin-1 encoding, then:

| Step | Rows / columns |
|---|---|
| Raw | 2,830,743 flows, 15 labels |
| Drop rows with NaN or ±inf (`Flow Bytes/s`, `Flow Packets/s`) | −2,867 |
| Drop exact duplicates (features + label) | −307,078 |
| Drop the duplicated `Fwd Header Length.1` column and 8 constant columns | 69 features remain |
| Clean | 2,520,798 flows |

The three Web Attack labels contain a mis-encoded byte in the original files (`Web Attack ï¿½ XSS`); the loader normalizes them to `Web Attack - XSS` and so on. 1,394 rows share identical features with a row of a different label; they are kept and reported.

The experiments use a stratified sample: up to 3,000 flows from each of the 14 attack labels (every flow for the rarer ones) plus an equal number of benign flows, 56,276 flows in total. The cleaned table is cached in `data/cicids2017/.cache/` on first load (about a minute).

Real per-label counts in the raw files, for reference:

| Label | Raw | After cleaning |
|---|---|---|
| BENIGN | 2,273,097 | 2,095,057 |
| DoS Hulk | 231,073 | 172,846 |
| PortScan | 158,930 | 90,694 |
| DDoS | 128,027 | 128,014 |
| DoS GoldenEye | 10,293 | 10,286 |
| FTP-Patator | 7,938 | 5,931 |
| SSH-Patator | 5,897 | 3,219 |
| DoS slowloris | 5,796 | 5,385 |
| DoS Slowhttptest | 5,499 | 5,228 |
| Bot | 1,966 | 1,948 |
| Web Attack - Brute Force | 1,507 | 1,470 |
| Web Attack - XSS | 652 | 652 |
| Infiltration | 36 | 36 |
| Web Attack - Sql Injection | 21 | 21 |
| Heartbleed | 11 | 11 |

## Synthetic API session dataset

```bash
python data/synthetic/generator.py                                   # original
python data/synthetic/generator.py --difficulty hard --output data/synthetic/sessions_hard.json
```

4,000 sessions (2,500 benign, 1,500 attack) across 5 organizations, seed 42. Each session becomes a 12-node endpoint graph with six node features: call count, mean duration, mean response size, maximum privilege level, write ratio, and a payload suspicion score (a count of tokens from a fixed lexicon).

| Attack type | Description | Organizations |
|---|---|---|
| lateral_movement | auth → user_read → user_write → admin_read → admin_write → data_export | 0, 2, 4 |
| slow_exfiltration | repeated bulk data_export calls | 0, 3, 4 |
| schema_recon | enumeration of most endpoints | 1, 2, 4 |
| sql_injection | SQL tokens in data_query parameters | 1, 3, 4 |

Organization 4 has all four attack types and is held out in the federated experiment.

**Known limitation of the original generator:** attack calls always carry tokens that benign calls never do, so the payload score alone separates the classes and every model with that feature reaches about 100%. The **hard** variant changes two things: 30% of benign sessions come from power users who add one to three admin, export or query calls with legitimate-looking admin, export, SQL or filter tokens, and each attack call carries ordinary payload tokens with probability 0.5 (camouflaged exfiltration calls also lose their 5× response size). Because the random stream changes, hard sessions do not align one-to-one with original sessions by index. The original generator's output is unchanged.
