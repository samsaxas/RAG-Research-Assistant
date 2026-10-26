# Real-world corpus

This folder holds the real document the project is evaluated on. The PDF itself is downloaded, not committed
by default (see `.gitignore`; remove that line if you prefer to commit the 1.8 MB file).

| | |
|---|---|
| Document | **NIST AI 100-1 — Artificial Intelligence Risk Management Framework (AI RMF 1.0)** |
| Publisher | U.S. National Institute of Standards and Technology (NIST), January 2023 |
| Length | 48 PDF pages (≈ 42 numbered pages + front matter) — one single document |
| Licence | U.S. government work, free of charge: <https://doi.org/10.6028/NIST.AI.100-1> |
| Why this one | A genuine, widely used governance standard: dense prose, long tables of numbered controls (GOVERN 1.1 … MANAGE 4.3), definitions, appendices — the kind of document people actually ask questions about. |

```bash
python scripts/fetch_real_doc.py      # downloads NIST.AI.100-1.pdf into this folder and verifies it
```

If your network blocks the download, save the PDF from the DOI link above and run
`python scripts/fetch_real_doc.py --file path/to/it.pdf`.
