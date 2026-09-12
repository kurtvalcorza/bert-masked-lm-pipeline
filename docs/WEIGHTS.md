# Weight provenance and DIMER hosting

- Upstream: `google-bert/bert-base-uncased`
- Immutable revision: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Weight format: SafeTensors (`model.safetensors`, 440449768 bytes, SHA-256 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`)
- Upstream weight license: Apache-2.0 (the snapshot carries the upstream `LICENSE` file)
- Local snapshot: `weights/bert-base-uncased/` with `dimer-base-manifest.json` (8 files, per-file bytes + SHA-256, `totalBytes` 441170446); the Git repository does not vendor the checkpoint.
- Load-time check: `stage_missing_files()` then `verify_snapshot()` in `src/bert_masked_lm_pipeline/pipeline.py` — the first refuses a manifest naming another model or revision and fetches only missing manifest entries when `allow_download=True`; the second re-hashes every manifest entry and refuses on any mismatch.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: `transformers==4.57.6` built-in `BertForMaskedLM` + `AutoTokenizer`, `trust_remote_code=False`, `local_files_only=True` from the snapshot directory; the checkpoint's pooler and next-sentence-prediction weights are not loaded. Hub download is opt-in and pinned to the revision above.
