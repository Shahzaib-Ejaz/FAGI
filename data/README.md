# Data

## CICIDS2017

Download from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

Place the file at:
```
data/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

The evaluation uses the Friday subset comprising 130,576 flows across seven classes:
- BENIGN (97,718)
- PortScan (15,890)
- DDoS (12,832)
- Bot (1,956)
- Web Attack Brute Force (1,507)
- Web Attack XSS (652)
- Web Attack SQL Injection (21)

## Synthetic API Session Dataset

Generate with:
```bash
python data/synthetic/generator.py --output data/synthetic/sessions_final.json
```

Or use the pre-generated file in `data/synthetic/sessions_final.json`.

The dataset comprises **4,000 sessions** across 5 organisational nodes with
overlapping aggregate statistics (write_ratio: benign 0.354±0.145, attack 0.357±0.158)
covering four attack types:

| Attack Type | Description | Org Coverage |
|-------------|-------------|--------------|
| lateral_movement | auth→user_read→user_write→admin_write→data_export escalation chain | Orgs 0, 2, 4 |
| slow_exfiltration | Repeated bulk data_export calls with camouflage | Orgs 0, 3, 4 |
| schema_recon | Systematic enumeration of all API endpoints | Orgs 1, 2, 4 |
| sql_injection | SQL tokens in data_query parameters | Orgs 1, 3, 4 |

Organisation 4 contains all four attack types and is used as the held-out
test node in Experiment 4 (federated generalisation).
