# tp-mt

toki pona <-> English / Chinese translation for on-device use (ilo sitelen), with atomic
`<tp:WORD>` tokens aligned 1:1 to the glyph classifier and an attention-linker output.

## pipeline

```
../tp-dataset/fetch_raw.sh                       # raw corpora, Linku, Tatoeba, Huang & Wu / Joo repos
uv run python ../tp-dataset/build_corpus.py      # corpus/{mono,parallel,dictionary}.jsonl + splits
uv run python align_corpus.py                    # eflomal links for the guided alignment loss
cd grammar && npm install && npm run build       # Huang & Joo nearley grammar bundle

uv run python train_mt.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/qwen2.5-0.5b
uv run python train_mt.py --model Qwen/Qwen2.5-1.5B-Instruct --out runs/qwen2.5-1.5b --grad-ckpt
uv run python train_mt.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/qwen2.5-0.5b-align --align-lambda 0.1 --grad-ckpt

uv run python eval_mt.py runs/<run>              # BLEU / chrF++ / grammaticality on the gold test split
uv run python eval_baselines.py                  # M2M-100 and GPT-2 Alpaca on the same split
uv run python compare.py                         # table incl. the contamination-safe post-2023 subset
uv run python linker.py runs/<run>               # attention-linker top-1 vs dictionary links
uv run python export_embeddings.py --run runs/<run>   # -> ../tp-dataset/embeddings/evaluate.py

uv run python merge.py runs/<run>                # standalone HF checkpoint; linker layers = best band in linker.json
cd ../torch-ios-conv && uv run python export_llm.py ../tp-mt/runs/<run>/merged   # CoreML, int4, stateful
```

## design

- **tokenizer** (`../tp-dataset/tp_tokenizer.py`): every `typst.toml` word is one added token,
  registered in `ocr-demo/class_mapping.json` order so `token_id = tp_offset + glyph_id`.
  Names go in `<cartouche> ... </cartouche>` through the base BPE. ~21% fewer tokens than raw BPE.
- **warm start** (`init_embeddings.py`): new rows = lipu Linku `ku_data`-weighted English gloss
  embeddings (+30% Chinese definition), blended with a ridge map from Huang & Wu's d=24 word2vec.
- **training** (`train_mt.py`): stage 1 trains only the new rows (1000 steps, transformer frozen);
  stage 2 LoRA r=32 on all projections + new rows. One unified chat format for tok<->en, tok<->zh,
  passage continuation and dictionary lookups (McDaniel 2023). Logits only at target positions.
- **guided alignment** (`align_loss.py`): KL between eflomal links and layer ⌊0.7L⌋ attention.
  Supervision is statistical; evaluation uses dictionary links, so the linker score is not circular.

## evaluation notes

- Test split: 1,556 gold tok-en and 309 gold tok-zh Tatoeba pairs, held out from training and
  from the monolingual corpus.
- Both prior baselines trained on public Tatoeba. `compare.py` therefore also reports the 551 / 243
  test pairs whose toki pona sentence was added to Tatoeba after 2023-05-02 (after both checkpoints).
  M2M-100 drops from 45.5 to 20.5 BLEU (tok->en) on that subset; our models do not.
- Grammaticality = share of outputs whose sentences all parse under Huang & Joo's (2025) nearley
  grammar (`grammar/`); gold references score 98.8%.

## results (gold test split; BLEU / chrF++ / grammatical %)

| run | en→tok | tok→en | zh→tok | tok→zh | linker top-1 |
|---|---|---|---|---|---|
| M2M-100 418M | 61.9 / 72.5 / 98.8 | 45.5 / 56.4 | – | – | – |
| GPT-2 Alpaca | 40.0 / 55.6 / 99.0 | 18.3 / 32.3 | – | – | – |
| Qwen2.5-0.5B | 47.8 / 65.1 / 98.2 | 26.6 / 44.4 | 42.6 / 61.7 / 98.4 | 20.7 / 16.9 | 54.4% (layers 0.8–1.0) |
| Qwen2.5-0.5B + guided alignment | 48.6 / 65.3 / 98.5 | 25.9 / 43.9 | 44.2 / 63.2 / 99.0 | 20.9 / 16.9 | **75.7%** (layer 16) |
| Qwen2.5-1.5B | 49.9 / 65.9 / 98.2 | 27.9 / 45.6 | 47.2 / 64.0 / 98.4 | 23.9 / 18.8 | 59.3% (layers 0.8–1.0) |

Post-2023 subset, en→tok / tok→en BLEU: M2M-100 42.4 / 20.5, GPT-2 35.3 / 11.9,
0.5B-align 45.2 / 25.9, 1.5B 48.2 / 25.6. Linker diagonal baseline: 26%.

Export sizes (int4): 0.5B 249 MB CoreML / 290 MB MLX; 1.5B 775 MB CoreML / 880 MB MLX.
The export pre-scales q and k by d^-1/4 each: Qwen2.5-1.5B's raw q·k overflows fp16.
MLX and CoreML outputs are marked `<tp:WORD>` tokens; decode with `TokiPonaTokenizer.decode`.
