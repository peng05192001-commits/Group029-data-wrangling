# Group029 — Multi-source retail data wrangling

A four-person FIT5196 coursework project: structurally parse JSON and XML, reconcile overlapping records, build six relational tables, validate their integrity, and explore customer, product, order, review and delivery patterns.

This is a **privacy-cleaned portfolio edition**, not the assessment submission archive. The original transformation code and EDA calculations are retained. Private source data and row-level outputs are not distributed, so a fresh clone can run the public tests but needs the original authorised data package for the complete pipeline.

## 项目简介

将两个来源的电商数据整合为六张关系表，完成文本清洗、日期/金额/布尔值标准化、重复记录协调、主外键及业务规则验证，并用 EDA 分析业务现象。

本项目由四位成员共同完成，不是仓库拥有者独立完成的项目。仓库拥有者负责 Member 1 的 customers / products 构建、标准化、关联检查和交接。其他成员负责订单、订单明细、评论、配送及整合分析；为保护隐私，这里以成员角色署名。开发和审查过程中使用过 AI 辅助，私人对话及签署声明不公开。

## What's included

| File / directory | Purpose |
|---|---|
| `Group029_solution.ipynb` / `.py` | Integrated parsing, transformation and validation workflow |
| `Group029_EDA.ipynb` | Reproducible EDA code and explanations |
| `Group029_EDA.pdf` | Original student-authored aggregate EDA report |
| `src/` | Member modules, retained under their existing import names |
| `Group029_text_functions.py` | Six reusable text-processing functions |
| `Group029_source_to_target_mapping.csv` | 111 field-lineage mappings |
| `templates/` | Referenced schema-mapping and text-test fixtures |
| `tests/` | Unit and integration tests |
| `scripts/test_public.py` | Tests that do not require private records |

## Quick start — no private data required

Use Python 3.12 and run commands from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/test_public.py
```

The public suite checks text cleaning, Unicode handling, extraction boundaries, scalar normalisation and mapping structure. It does **not** replace the full source-data integration tests.

## Reproduce the complete workflow

Obtain the original Group029 data package through an authorised channel. Do not upload it here. Restore this layout locally:

```text
raw_package/
  A1_manifest.json
  public_data_dictionary.csv
  raw_input/
    Group029_commerce.json
    Group029_operations.xml
```

Then run:

```bash
python Group029_solution.py
python -m jupyter nbconvert --execute --to notebook --inplace Group029_EDA.ipynb
python -m unittest discover -s tests -v
```

Alternatively open both notebooks in VS Code/Jupyter, select the installed environment, and **Restart and Run All**: solution first, EDA second. The pipeline checks the original manifest, creates the six CSVs in `outputs/`, and the EDA notebook creates figures locally. Public notebooks intentionally contain no cached output, execution paths or attachment metadata. Their report findings refer to the original dataset, not to newly generated demonstration data.

## Privacy and provenance

- Excluded: AI conversation exports and declarations, signatures, student numbers, personal local paths, raw JSON/XML, record-level CSVs, cached notebook outputs, temporary files and old submission ZIPs.
- Preserved: original code, field mapping, tests, aggregate report and role-based attribution. Course handouts and marking documents are not included.
- `.gitignore` prevents routine re-addition of private inputs and generated outputs. Review staged files before every push: ignoring files is not a substitute for a privacy check.
- No open-source licence has been added on behalf of the group or the course. Public visibility alone does not grant a general reuse licence. Use the project to understand methods, not to submit someone else's coursework.

## Validation boundary

The original data-backed workflow was previously rerun locally: both notebooks completed and 46 tests passed. The portfolio preparation preserves source code and clears display outputs only. Public test and privacy-audit results are recorded in `PUBLIC_RELEASE.md`; full results require the authorised input package above.
