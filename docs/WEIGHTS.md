# Weight provenance and DIMER hosting

- Upstream: `google-bert/bert-base-uncased`
- Immutable revision: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Weight format: SafeTensors (`model.safetensors`, 440449768 bytes, SHA-256 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`)
- Upstream weight license: Apache-2.0 (the snapshot carries the upstream `LICENSE` file)
- Local snapshot: `weights/bert-base-uncased/` with `dimer-base-manifest.json` (8 files, per-file bytes + SHA-256, `totalBytes` 441170446); the Git repository does not vendor the checkpoint.
- Load-time check: `stage_missing_files()` then `verify_snapshot()` in `src/bert_masked_lm_pipeline/pipeline.py` — the first refuses a manifest naming another model or revision and fetches only missing manifest entries when `allow_download=True`; the second re-hashes every manifest entry and refuses on any mismatch.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: `transformers==4.57.6` built-in `BertForMaskedLM` + `AutoTokenizer`, `trust_remote_code=False`, `local_files_only=True` from the snapshot directory; the checkpoint's pooler and next-sentence-prediction weights are not loaded. Hub download is opt-in and pinned to the revision above.

## Adaptation corpus (tutorial data, not weights)

- Corpus: SciTLDR-A paper abstracts (Cachola et al., EMNLP Findings 2020), `allenai/scitldr` at commit `5ccad9c00a60ad75c9e04abf7f27d0f53f983b20`, licence Apache-2.0.
- Files: `SciTLDR-Data/SciTLDR-A/train.jsonl` (3,155,015 bytes, SHA-256 `b222771d387be585cfdf5ae957b36757138415a352e0a3e3b23f73f87c3b1119`), `dev.jsonl` (1,124,865 bytes, `3191fa98ccc09521332b7a1cd63b1930be4e8df125a235ccd31e40329709525e`), `test.jsonl` (1,204,107 bytes, `fb42dd6cd4f4a1928ae8a01a189456fbfe994a07e938bd49f68653933f6503c9`), fetched by `samples.fetch_corpus` from `raw.githubusercontent.com` over HTTPS at run time into the git-ignored `weights/scitldr/` cache and refused on any byte or SHA-256 mismatch. Only the `source` (abstract sentences) field is read; titles and TLDRs are ignored.
- Sample: `build_sample_dataset(seed=42)` draws 300 / 50 / 100 `{id, text}` records from the release's own train/dev/test members (abstracts of 200..2,000 characters, de-duplicated). The repository redistributes none of the corpus; DIMER hosting of the weights is unaffected.
- Adapter artifacts written by the tutorial (`outputs/bert_masked_lm_adapter/`, `org.valcorza.bert-base-uncased.adapter.v1`, about 113 MB) carry only the trained encoder-layer tensors and a manifest naming the base `model.safetensors` digest; they are outputs, not hosted weights.
