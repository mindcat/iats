# Autonomous Agent Execution Plan: `mindcat/iats` Pipeline Upgrade

**Project:** Sitelen Pona Optical Character Recognition, Layout Segmentation, Custom Embedding Space, and Edge Neural Machine Translation (`ilo sitelen` backend)  
**Repository Target:** `mindcat/iats`  
**Deployment Target:** Local Edge Execution on iOS (< 1.5 GB total footprint; CoreML for Vision models + CoreML Stateful / MLX Swift for LLM with Attention Linker output)

---

## 0. Executive Summary & Optimal Execution Order

This document provides the complete technical context, theoretical grounding from literature, and step-by-step implementation instructions for an autonomous coding agent.

### Recommended Execution Order: **Task 1** $\rightarrow$ **Task 3** $\rightarrow$ **Task 2**

As hypothesized, **Task 3 (Corpus Collation, Custom Tokenizer, & MT Finetuning) must precede Task 2 (Segmentation & Heuristic Grouping Dataset Generation)** for three structural reasons:

1. **Authentic Multi-Sentence & Short-Phrase Layouts:** Task 2 requires rendering realistic multi-line text-box paragraphs alongside independent stray single glyphs and short 2–4 word captions/sentences to train and evaluate the YOLOv8 detector and the spatial clustering + line/paragraph heuristic grouping algorithm. Pulling cleaned, length-stratified passages from the **Task 3 collated corpus (`poki Lapo`, `ilo Muni`, `tok.wikipedia`, `lipu tenpo`)** ensures the synthetic page layouts reflect true Toki Pona word-length distributions, particle density (`li`, `e`, `pi`, `la`), and proper-noun cartouche frequencies.
2. **Unified Glyph-to-Token Indexing:** Updating the CNN classifier's class list in **Task 1** from `tp-dataset/categories/typst.toml` establishes the canonical `glyph_id` vocabulary (plus the negative `not_sitelen_pona` reject class). Designing the **Task 3 Custom Tokenizer** immediately after Task 1 allows a strict 1:1 index alignment between the visual CNN classes and the LLM's dedicated `<tp:WORD>` token IDs.
3. **Training Compute Overlapping:** Task 3 involves multi-model trial runs (continued pretraining + instruction/translation finetuning). Starting data preparation and training runs for Task 3 early allows the agent to build and verify the Task 2 synthetic page generator and heuristic grouping script in parallel.

---

## 1. Repository & Literature Context (Must-Read for Agent)

### 1.1 Existing `mindcat/iats` Workspace Structure

* `tp-dataset/categories/typst.toml`: **Canonical source of truth** for the expanded Sitelen Pona wordlist/glyph classes to be used in both Task 1 (CNN Classifier) and Task 3 (Custom Tokenizer).
* `ocr-demo/`: Contains the CNN glyph classifier pipeline (`dataset.ipynb`, `train.ipynb`, `test.ipynb`) and 13 Sitelen Pona `.ttf`/`.otf` fonts (`Lilintawa.ttf`, `fairfax-pona.ttf`, `kasi linja tan kasi Lija.ttf`, `linja-sike.otf`, `linjalipamanka-normal.ttf`, `loje meli.ttf`, `nasin-nanpa.otf`, `sitelen Lenluwi.ttf`, `sitelen-Tojoko.ttf`, `sitelen-ante-en-ijo-ona-utala-Esi-tan-mu-Esitense.ttf`, `sitelen-pi-pan-Nuto.ttf`, `sitelen-pona-Elopa-pi-ma-Sapu.ttf`, `sitelenselikiwenasuki.ttf`).
* `torch-ios-conv/`: Scripts (`main.py`, `ocr.ipynb`) for converting PyTorch checkpoints to `.mlpackage` via `coremltools` for on-device iOS execution.
* `glyph-seg/`: Workspace for YOLOv8 glyph/cartouche bounding-box detection (`[x, y, size, cart]`) and heuristic line/paragraph/text-box grouping (currently empty of a working grouping script; must be implemented from scratch in Task 2).
* `m2m/`, `tp-dataset/`, `tp-gpt2/`: Existing translation experiments (M2M-100, GPT-2, dataset scripts).

### 1.2 Key Findings from Provided Literature (Baselines & Inspiration)

1. **Wangeman & Khan (2026) — *CMPE-679 Baseline*:**
   * **CNN Classifier:** Takes $28 \times 28$ grayscale patches. Architecture: `Conv2d(1, 32, k=3)` + ReLU + MaxPool $\rightarrow$ `Conv2d(32, 64, k=3)` + ReLU + MaxPool $\rightarrow$ Flatten $\rightarrow$ `Linear(256)` + ReLU + Dropout $\rightarrow$ `Linear(N_classes)` + Softmax. Trained for 10 epochs on MPS using 670,000 synthetic images in an `.npz` archive. Achieved $98.30\%$ validation accuracy, $0.0546$ validation loss, and a $1.7\text{ MB}$ CoreML footprint.
   * **MT Findings:** Evaluated `TranslateGemma`, `Qwen-MT`, and `Gemma 4`. Unquantized 27B/29B Gemma 4 was strongest on GPU, but under heavy edge quantization, **Chinese-centric analytical models (Qwen family)** demonstrated superior contextual retention per parameter compared to Gemma-3-based models because Chinese and Toki Pona share a strictly **analytical, isolating morphology**.

2. **Huang & Wu (2026) — *Examining the Limits of Word2Vec with Toki Pona (SCiL 2026)*:**
   * **Data Scale:** Sourced $1.42\text{M}$ sentences ($7.95\text{M}$ tokens) filtered via `sona-toki` (Danielson, 2025).
   * **Critical Finding on "Linguistic Noise" (Loanwords & Proper Nouns):** Approximately $23\%$ of natural Toki Pona sentences contain non-core tokens ($113,663$ named entity occurrences across $519$ types like *Sonja*, *Inli*, *Jutu*; $55,062$ nonstandard/neologism occurrences like *lanpan*, *kijetesantakalu*, *soko*, *misikeke*, *tonsi*). Training on the **Full Corpus (retaining named entities and neologisms)** yielded a significantly higher semantic category centroid cosine similarity ($\mu = 0.879$) than the **Pure Toki Pona** corpus ($\mu = 0.817$). Named entities (e.g., *Jutu* co-occurring with *lipu*, *ilo*, *sitelen*) act as **structural semantic bridges** that tighten polysemous clusters without distorting the underlying cross-linguistic geometry.
   * **Intrinsic Dimensionality:** Using Yin & Shen (2018), the optimal standalone embedding dimensionality for Toki Pona's core vocabulary is $d = 24$, and the 27 semantic supersense categories (108 content words, 125 assignments) provide a rigorous quantitative benchmark (`Centroid Retrieval Rate`, `Agglomerative Silhouette Score`, and `English Google-News Representational Similarity Analysis (RSA)`).

3. **Huang & Joo (2025) — *Language Change and Variation in Toki Pona (arXiv:2508.10246)*:**
   * **Corpora:** Contrasted informal chat (`ma pona pi toki pona` Discord) with formal long-form prose (`poki Lapo`, `github.com/kulupu-lapo/poki`).
   * **Syntactic Fluidity & Earley Parser:** Because every Toki Pona content word is fluid across `NOUN`, `MOD` (modifier), `IVERB` (intransitive), and `TVERB` (transitive before `e`), plus closed-class `PREP` (`lon`, `tawa`, `tan`, `kepeken`, `sama`), `PREVERB` (`wile`, `kama`, `awen`, `ken`, `sona`, `lukin`), and `PART` (`li`, `e`, `pi`, `la`, `o`, `en`, `anu`, `seme`), context dictates meaning. Notably, body parts (`luka`, `uta`) are increasingly used as transitive verbs ($\sim 15\%$ of occurrences by 2024), and conversational data has high standalone interjection rates (`pona`, `wawa`) compared to `poki Lapo`.

4. **McDaniel (2023) — *Toki Pona Language Models*:**
   * Demonstrated that training on raw next-token prediction or arbitrary delimiter strings (`>>>>>`) causes small models to hallucinate endless chains of grammatically valid but semantically detached Toki Pona.
   * **Key Takeaway:** Framing **both** monolingual generation ("Write a story/article in Toki Pona about X", "Continue the following Toki Pona passage") and bilingual translation within a **unified instruction-following template** (and including **Toki Pona** $\leftrightarrow$ **Chinese** alongside **Toki Pona** $\leftrightarrow$ **English** pairs from Tatoeba) dramatically stabilizes semantic adherence.

---

## 2. Phase 1 (Task 1): Re-running & Updating the CNN Glyph Classifier

### Objective

Parse the updated character wordlist from `tp-dataset/categories/typst.toml`, add a dedicated negative/reject class (`not_sitelen_pona`) containing non-Toki Pona characters (Latin, Chinese, Korean, symbols), regenerate the $28 \times 28$ `.npz` dataset across all valid fonts in `ocr-demo/fonts/`, retrain the CNN classifier in `ocr-demo/`, and export the updated `.mlpackage` via `torch-ios-conv/`.

### Step-by-Step Agent Instructions

1. **Parse `tp-dataset/categories/typst.toml` & Audit Font Coverage:**
   * Load `tp-dataset/categories/typst.toml` using Python's `tomllib` to extract the complete list of target Sitelen Pona words/characters.
   * Map each word to its corresponding UCSUR codepoint (`U+F1900`–`U+F19FF`) and/or ligature representation used by the 13 fonts in `ocr-demo/fonts/`.
   * **Font Cmap Validation (Critical):** Using `fontTools.ttLib.TTFont`, programmatically check which of the 13 font files in `ocr-demo/fonts/` actually contain a valid glyph outline (non-empty `glyf`/`CFF ` contour and not mapped to `.notdef` / blank box) for each word in `typst.toml`. Only render a given character using fonts that support it.
   * **Negative Class (`not_sitelen_pona`):** Add 1 additional class at the end of the class list (`class_id = N_TP_WORDS`) for unrecognized shapes. Sample characters from system fonts covering:
     * Basic Latin (`A-Z`, `a-z`, `0-9`, ASCII punctuation)
     * Common CJK Unified Ideographs (Chinese characters)
     * Hangul syllables (Korean characters)
     * Geometric/mathematical symbols and random stroke scribbles
     Generate $\sim 2\times$ the per-class sample count for `not_sitelen_pona` ($\sim 10,000$ patches) so the classifier achieves a low false-positive rate on background text or non-Sitelen Pona camera clutter.

2. **Regenerate Synthetic Dataset (`.npz`):**
   * Convert the dataset generation logic in `ocr-demo/dataset.ipynb` into a clean CLI script `ocr-demo/generate_dataset.py` (and sync back to `dataset.ipynb`).
   * Keep the proven augmentation pipeline at $28 \times 28$ grayscale resolution: random rotation ($\pm 15^\circ$), affine scale/translation/shear, stroke width/dilation/erosion jitter, Gaussian blur/noise, and polarity inversion (white-on-black and black-on-white for both PencilKit and camera crops).
   * Save compressed `train_data.npz`, `val_data.npz`, and a canonical `class_mapping.json` mapping `index -> {word, ucsur_hex, category}` so downstream tasks (Task 3 & iOS app) read a single source of truth.

3. **Update CNN Architecture Head & Retrain:**
   * In `ocr-demo/train.ipynb` (and `ocr-demo/train.py`), update the output layer `nn.Linear(256, N_TOTAL_CLASSES)` where `N_TOTAL_CLASSES = len(typst_wordlist) + 1`.
   * Train on `mps` (`torch.device("mps")`) for 10–15 epochs using AdamW (`lr=1e-3` with CosineAnnealingLR).
   * Verify overall validation accuracy $\ge 98.0\%$ and inspect per-class recall on the newly added characters and the `not_sitelen_pona` reject class.

4. **CoreML Export via `torch-ios-conv/`:**
   * Update `torch-ios-conv/main.py` and `torch-ios-conv/ocr.ipynb` to load the new checkpoint and `class_mapping.json`.
   * Export to `SitelenPonaClassifier.mlpackage` using `coremltools` with classifier metadata/class labels baked into the `.mlpackage` spec. Verify the model footprint remains $\sim 1.7\text{–}2.0\text{ MB}$.

---

## 3. Phase 2 (Task 3): Custom Toki Pona Tokenizer, Embeddings, & Edge MT Finetuning

### 3.1 On-Device Runtime Architecture: CoreML vs. MLX Swift

You noted that `torch-ios-conv` (CoreML) worked great for the CNN classifier and asked whether CoreML or MLX Swift is better for the translation LLM:

* **Keep CoreML (`torch-ios-conv`) for Vision Models (Model 1 YOLOv8 & Model 2 CNN):** CoreML automatically dispatches convolutions to the Apple Neural Engine (ANE) with ultra-low power and a $1.7\text{ MB}$ footprint. Do not change this.
* **Why to Use CoreML (`coremltools` Stateful `.mlpackage` with `int4` palettization) OR `MLX Swift` for the LLM (Model 3):**
  * **Option 3A — CoreML Stateful `.mlpackage` (Unified Pipeline):** Modern `coremltools` (v8+) supports exporting PyTorch Transformers with `ct.State` KV-caching and 4-bit weight palettization (`ct.optimize.torch.quantization`). **Crucial advantage for your Attention Linker:** When exporting the PyTorch `forward()` pass in `torch-ios-conv/`, you can explicitly return a tuple `(logits, source_attention_weights)` as named CoreML outputs! That means your Swift code receives the exact $M$-length attention vector over the source sentence on every single token step with zero custom C++/Metal shaders.
  * **Option 3B — MLX Swift:** `mlx-swift` is faster at raw token/sec decoding for Qwen/Gemma on iPhone 16/17 GPUs and loads HuggingFace safetensors directly, **but** standard `mlx-swift-examples` LLM runners do not expose internal layer attention weights by default—you have to modify the Swift `TransformerBlock` forward pass to return `scores` before `SDPA`.
  * **Recommendation for Agent:** Build the PyTorch wrapper so it can export **both** (1) a 4-bit palettized **CoreML `.mlpackage`** in `torch-ios-conv/export_llm.py` that outputs `(logits, alignment_probs)` directly for plug-and-play Xcode integration, and (2) an **MLX 4-bit quantized folder** for benchmarking speed.

### 3.2 Custom Toki Pona Tokenizer & Vocabulary Extension (Preserving English Embeddings)

As confirmed, **preserving the base model's pretrained English (and Chinese) embeddings is strongly preferred** because small low-resource finetunes rely heavily on the pretrained semantic geometry of English to translate accurately.

**How the Agent Must Implement Vocabulary Extension (`Option A`):**

1. **Dedicated Atomic Toki Pona Tokens (`<tp:WORD>`):**
   * Load the exact wordlist from `tp-dataset/categories/typst.toml` (matching Task 1).
   * Instead of letting the base BPE tokenizer split Toki Pona words (`kijetesantakalu` $\rightarrow$ 4 subwords) or letting short Toki Pona words (`li`, `e`, `pi`, `la`, `o`, `en`, `jan`, `sin`, `pan`) share token IDs with English/Pinyin subwords (`li` in *library*, `pan` in English *pan*), register every word in `typst.toml` as a distinct atomic token: `<tp:toki>`, `<tp:pona>`, `<tp:li>`, `<tp:e>`, etc., plus structural tokens `<cartouche>` and `</cartouche>`.
   * Write a lightweight pre-tokenizer/wrapper (`tp-dataset/tp_tokenizer.py`) that converts Toki Pona text into `<tp:WORD>` tokens while leaving proper nouns inside `<cartouche> ... </cartouche>` to be tokenized phonetically by the base tokenizer (e.g., `jan Sonja` $\rightarrow$ `<tp:jan> <cartouche> Son ja </cartouche>`).

2. **Semantic Warm-Start Initialization (No Random Vectors!):**
   * Resize the LLM's token embedding matrix and LM head (`model.resize_token_embeddings(len(tokenizer))`).
   * For every new `<tp:WORD>` token, initialize its row in `embed_tokens` and `lm_head` as the **weighted average of the pretrained base model's embeddings for its English (and Chinese) dictionary definitions** from `lipu Linku` (`linku.la`), refined via Procrustes alignment from our standalone Toki Pona embedding space.
   * During Stage 1 of training, freeze the transformer blocks for 1,000 steps and train *only* the newly added `<tp:WORD>` rows in `embed_tokens` and `lm_head` so they snap cleanly into the base model's native manifold without degrading English fluency.

3. **Standalone Research Embedder & Visualization Suite (`tp-dataset/embeddings/`):**
   * Train the standalone Toki Pona Word2Vec/FastText ($d = 24$, optimal per Huang & Wu 2026) and extract the LLM's learned `<tp:WORD>` embeddings ($d_{\text{model}}$).
   * Automatically run the **Huang & Wu (2026) Evaluation Suite**:
     * 27-Category Centroid Retrieval Rate (target: $\ge 124/125$)
     * Unsupervised Agglomerative Clustering Silhouette Score (target: $> 0.267$)
     * UMAP 2D semantic projection plots (`Full Corpus` vs. `Pure TP` vs. `LLM Contextual`)
     * $27 \times 27$ Cross-Linguistic Representational Similarity Analysis (RSA) heatmap comparing Toki Pona against English embeddings.

### 3.3 Corpus Collation & Low-Resource Training Strategy

The agent will create `tp-dataset/build_corpus.py` combining:

1. **Monolingual Data (`ilo Muni` + `poki Lapo` + `tok.wikipedia`):**
   * Fetch the `ilo Muni` database (`https://ilo.muni.la/blog/about/` / `gregdan3` SQLite dump containing $1.4\text{M}+$ `sona-toki`-filtered sentences), `poki Lapo` (`github.com/kulupu-lapo/poki`), and `tok.wikipedia.org` articles.
   * **Keep Adapted Loanwords & Named Entities:** Per Huang & Wu (2026), retain sentences containing proper names (`jan Sonja`, `toki Inli`, `ilo Jutu`) and community words in `typst.toml`, as they act as semantic anchors that increase category centroid similarity ($\mu = 0.879$ vs. $0.817$).
2. **Parallel Data (`Tatoeba` EN-TP & ZH-TP + Long-Form Texts):**
   * Extract all Toki Pona $\leftrightarrow$ English and Toki Pona $\leftrightarrow$ Mandarin Chinese pairs from Tatoeba, plus parallel biblical chapters, fables, and `lipu tenpo` translations.
   * **Synthetic Back-Translation & Dictionary Grounding:** Use `lipu Linku` definitions and compound phrases (`ilo pi ante toki` = `machine translator`) plus back-translated `poki Lapo` paragraphs to create multi-sentence context-aware translation pairs (solving the single-sentence context limitation noted in CMPE-679 Sec I).
3. **Comparative Trial Runs:**
   * Run standardized 3-epoch trial finetunes (using LoRA + trainable `embed_tokens`/`lm_head`) across:
     1. **`Qwen2.5-1.5B-Instruct` (or `Qwen3-1.7B`):** Primary Chinese-analytical candidate ($\sim 0.9\text{ GB}$ at 4-bit).
     2. **`Gemma-4-2B-it` (or text-stripped `Gemma` / `TranslateGemma-2B`):** Primary Google candidate ($\sim 1.1\text{ GB}$ at 4-bit).
     3. **`Qwen2.5-0.5B-Instruct`:** Ultra-fast edge candidate ($\sim 0.35\text{ GB}$ at 4-bit).
   * Compare on a held-out test set using **BLEU**, **chrF++**, **COMET**, and **Earley Parser Grammaticality Rate** (`nearley` CFG from Huang & Joo 2025).

### 3.4 Attention Map Extraction (Pseudo Word-to-Word Linker)

To let the iOS app highlight which source words influenced each generated target token:

* **Architecture:** Wrap the finetuned model in `AttentiveTranslatorWrapper(nn.Module)`. During each decoding step for target token $t_j$, extract the self-attention weights from the middle-to-upper semantic alignment layers (e.g., mean across heads of layers $\lfloor 0.6 L \rfloor$ to $\lfloor 0.8 L \rfloor$), slice **only** the columns corresponding to the source token indices $[s_1, \dots, s_M]$, mask out `<bos>` and prompt template tokens (eliminating attention sinks), and normalize over the source words:
  $$a_{j, i} = \frac{\sum_{l \in \mathcal{L}_{\text{align}}} \sum_{h=1}^{H} \text{Attn}^{(l,h)}(t_j, s_i)}{\sum_{k=1}^{M} \sum_{l \in \mathcal{L}_{\text{align}}} \sum_{h=1}^{H} \text{Attn}^{(l,h)}(t_j, s_k)}$$
* **Guided Alignment Supervision:** During finetuning, compute word alignments on the parallel corpus using `fast_align` / `eflomal` + `lipu Linku` dictionary matches, and add an auxiliary KL-divergence loss $\lambda \mathcal{L}_{\text{align}}$ ($\lambda = 0.1$) on layer $\lfloor 0.7 L \rfloor$ so the attention distribution is explicitly trained to produce sharp, human-interpretable word-to-word links.

---

## 4. Phase 3 (Task 2): Synthetic Dataset & From-Scratch Heuristic Grouping (`glyph-seg/`)

### Objective

Since `glyph-seg/` currently has no working grouping script, the agent must implement **both**:
1. **`glyph-seg/heuristic_grouping.py`:** A deterministic geometric layout parser that takes detected Sitelen Pona glyphs/cartouches (`[x, y, w, h, cartouche]`), separates **structured multi-line text boxes** from **independent stray single glyphs or short standalone sentences** in the same image, and reconstructs lines, paragraphs, and reading order.
2. **`glyph-seg/generate_seg_dataset.py`:** A synthetic scene generator that creates realistic images containing multi-line text boxes alongside independent stray glyphs/short captions (just like the synthetic data approach that generalized nearly perfectly in Task 1) to train YOLOv8 and validate/tune the heuristic grouping algorithm.

### Step-by-Step Agent Instructions

#### 4.1 Designing the From-Scratch Heuristic Grouping Algorithm (`glyph-seg/heuristic_grouping.py`)

Given a set of detected glyphs $G = \{g_1, \dots, g_N\}$ where each glyph $g_i = (x_i, y_i, w_i, h_i, c_i)$ has center $(x_i, y_i)$ and characteristic scale $s_i = \max(w_i, h_i)$ (and $c_i \in \{0, 1\}$ indicates a cartouche):

1. **Step 1 — Cartouche Consolidation:**
   * Any glyphs whose centers fall inside a detected `cartouche` bounding box $B_{\text{cart}}$ are linked as an ordered internal sequence (sorted left-to-right along the cartouche's major axis), and the cartouche itself is treated as a single macro-glyph node with height $h_{\text{cart}}$ for line-tracing purposes.

2. **Step 2 — Macro-Block Separation (Text Boxes vs. Independent Stray Glyphs / Short Sentences):**
   * Compute a scale-normalized anisotropic distance between every pair of nodes $(i, j)$ using the local mean glyph size $\bar{s}_{ij} = \frac{s_i + s_j}{2}$:
     $$d_{\text{layout}}(i, j) = \sqrt{\left(\frac{\Delta x_{ij}}{\gamma_x \bar{s}_{ij}}\right)^2 + \left(\frac{\Delta y_{ij}}{\gamma_y \bar{s}_{ij}}\right)^2}$$
     where $\gamma_x \approx 1.8$ (horizontal kerning tolerance) and $\gamma_y \approx 1.6$ (vertical line-spacing tolerance).
   * Build a connectivity graph where an edge $(i, j)$ exists if $d_{\text{layout}}(i, j) \le 1.0$ **and** their sizes are compatible ($\frac{\max(s_i, s_j)}{\min(s_i, s_j)} < 1.75$, preventing huge background/title glyphs from merging into small body text).
   * Extract **Connected Components (Regions/Blocks)** $\{R_1, R_2, \dots, R_K\}$. This immediately isolates:
     * **Isolated Single Glyphs** ($|R_k| = 1$)
     * **Independent Short Sentences / Captions** ($|R_k| \in [2, 6]$, single row, spatially separated from other blocks)
     * **Structured Text Boxes / Paragraphs** (dense multi-row rectangular/polygonal blocks).

3. **Step 3 — Line Tracing ("Drawing Lines Through Sitelen Pona") Within Each Region $R_k$:**
   * Estimate the dominant local baseline angle $\hat{\theta}_k$ of region $R_k$ using PCA / RANSAC on horizontal nearest-neighbor vectors $(\Delta x, \Delta y)$ where $|\Delta y| < 0.6 \bar{s}$ and $\Delta x \in [0.5 \bar{s}, 2.2 \bar{s}]$.
   * Rotate the coordinates of $R_k$ by $-\hat{\theta}_k$ into canonical horizontal alignment $(x', y')$.
   * **Iterative Line Sweep:** Group glyphs into **Rows** by projecting horizontal rays / 1D kernel density estimation along $y'$ (merging glyphs whose vertical intervals $[y'_i - 0.4 s_i, y'_i + 0.4 s_i]$ overlap by $> 40\%$). Fit a smooth piecewise-linear spline through the centers of each row to handle slight camera/page curvature.
   * Sort glyphs within each row by $x'$ (left-to-right).

4. **Step 4 — Paragraph Splitting & Reading Order Output:**
   * Within a multi-line Text Box $R_k$, sort rows top-to-bottom by mean $y'$.
   * Compute inter-row vertical gaps $\Delta y'_{r, r+1}$ and left-indent shifts $\Delta x'_{r, r+1}$. Split into separate **Paragraphs** whenever $\Delta y'_{r, r+1} > 1.4 \cdot \text{median}(\Delta y')$.
   * Output a structured hierarchy JSON:
     ```json
     {
       "regions": [
         {
           "region_id": 0,
           "type": "text_box | short_sentence | single_glyph",
           "bbox": [x_min, y_min, x_max, y_max],
           "paragraphs": [
             {
               "lines": [
                 [{"glyph_id": 12, "x": 104, "y": 210, "in_cartouche": false}, ...]
               ]
             }
           ]
         }
       ]
     }
     ```

#### 4.2 Synthetic Segmentation & Layout Dataset Generator (`glyph-seg/generate_seg_dataset.py`)

To mirror the synthetic data generation success of Task 1, generate $20,000+$ composite training scenes ($640 \times 640$ and $1024 \times 1024$) containing a randomized mix of:
1. **1 to 2 Structured Multi-Line Text Boxes:** 2 to 6 lines of Toki Pona text sampled from `poki Lapo` / `ilo Muni`, rendered inside a virtual bounding box (with optional visible border, chat bubble, or paper card background) with word wrapping, left/center alignment, and cartouches.
2. **1 to 4 Independent Short Sentences or Stray Single Glyphs:** Placed elsewhere on the canvas (different font size, angle, or margin location) to train and test the algorithm's ability to keep them independent from the main text box.
3. **Distractor Non-Sitelen Pona Text/Shapes:** Occasional Latin/CJK words or UI icons (labeled as background for YOLO and `not_sitelen_pona` for the CNN).
4. **Realistic Camera Augmentations:** Perspective homography warp, slight barrel/page curl distortion, lighting shadows, motion blur, and JPEG compression.

---

## 5. Verification & Benchmarking Suite

| Component | Metric / Benchmark | Target / Baseline to Beat |
| :--- | :--- | :--- |
| **Task 1: CNN Classifier** | Validation Accuracy on `typst.toml` classes + `not_sitelen_pona` | $\ge 98.3\%$ accuracy; $< 1\%$ false positive rate on Latin/CJK/symbols; $< 2.0\text{ MB}$ `.mlpackage` |
| **Task 3A: Custom Embedder** | Huang & Wu (2026) 27-Category Centroid Retrieval & Silhouette Score | Beat or match $124/125$ category retrieval & $\ge 0.267$ Silhouette; generate UMAP & $27\times 27$ English RSA plots |
| **Task 3B: Edge MT Model** | BLEU, chrF++, COMET, & Grammatical Validity (`nearley` parser pass rate) | Beat M2M-100 & GPT-2 Alpaca baselines across `Qwen2.5-1.5B` / `Gemma-4-2B`; $>95\%$ grammatical parse rate; $< 1.2\text{ GB}$ 4-bit |
| **Task 3C: Attention Linker** | Top-1 Source Word Alignment Accuracy | $> 85\%$ Top-1 accuracy linking generated target tokens to ground-truth source words; exported directly as second CoreML output |
| **Task 2: Seg & Grouping** | YOLO mAP@50, Region Separation F1, & Reading-Order Kendall's $\tau$ | YOLO mAP@50 $> 0.96$; $> 98\%$ F1 separating Text Boxes from independent short sentences/single glyphs; Kendall's $\tau > 0.99$ |