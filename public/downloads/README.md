# NISQA v2.0 + 1D Temporal Dilated CNN Hybrid Audio QA Pipeline (Up to 48 kHz)

A **100% Offline, Non-Intrusive (Single-Ended, Zero-Reference)** production Audio Quality Assurance (QA) pipeline designed to segregate mono audio recordings up to **48 kHz** (`.wav`, `.flac`, `.mp3`) into **OK (Good)** vs. **NG (Bad)** and sub-classify NG defects into **NG_Noise**, **NG_Cut**, **NG_Wobble**, and **NG_Rep** with **25ms peak defect timestamp localization**.

---

## 1. Why NISQA v2.0 + 1D Dilated CNN is the Optimal Architecture

| Tool / Standard | Reference Needed? | Max Sample Rate | Native Python Batch | License | Verdict for 48 kHz Offline QA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NISQA (v2.0) + 1D Dilated CNN** | **Non-Intrusive (No Reference)** | **48 kHz (Fullband)** | **Yes (PyTorch + Keras)** | **Open-Source (MIT)** | **WINNER: 5 perceptual dimensions + temporal rhythm & repetition gate** |
| PESQ (ITU-T P.862) | Intrusive (Needs golden ref) | 16 kHz (Wideband) | Yes | Proprietary ITU | Disqualified: Needs reference & blind > 8 kHz |
| ViSQOL (v3) | Intrusive (Needs golden ref) | 48 kHz / 16 kHz | Yes | Apache 2.0 | Disqualified: Strictly full-reference |
| ITU-T P.563 | Non-Intrusive | 8 kHz (Narrowband) | External C binary | Proprietary ITU | Disqualified: Downsamples to 8 kHz (erases 83% of 48 kHz spectrum) |
| GL Comm. VQT (P.563) | Non-Intrusive | 8 kHz | Proprietary SDK | Commercial Hardware | Disqualified: Uses 8 kHz P.563 engine |
| Sevana AQuA | Intrusive | 48 kHz | CLI / SDK | Commercial License | Disqualified: Needs reference waveform |
| Sevana PVQA (PQUA) | Non-Intrusive | 8 kHz / 16 kHz | C++ / Python SDK | Commercial License | Sub-optimal: Tuned for narrow/wideband VoIP |
| SpeechMetrics | Mixed (Mostly Intrusive) | 16 kHz (MOSNet/SRMR) | Yes | Open-Source | Sub-optimal: 16 kHz limit; SRMR measures reverb, not cuts/wobbles |

---

## 2. Two-Stage Cascaded Hybrid Architecture

```text
           [ Input Audio: Up to 48 kHz Mono (.wav / .flac / .mp3) ]
                                      │
      ┌───────────────────────────────┴───────────────────────────────┐
      ▼                                                               ▼
[ Path A: NISQA v2.0 (Native 48 kHz) ]          [ Path B: 63-Feature Physics Stack ]
Outputs 5 Perceptual Dimensions (1.0 - 5.0):    1D Dilated CNN (1,2,4,8) + AttentionLayer
• mos_pred  (Overall Quality)                   Outputs:
• noi_pred  (Noisiness)                         • 5-Way Softmax Probabilities
• dis_pred  (Discontinuity)                     • Temporal Attention Heatmap (25ms)
• col_pred  (Coloration)                        • Acoustic Self-Similarity Score
• loud_pred (Loudness)                                                │
      │                                                               │
      └───────────────────────────────┬───────────────────────────────┘
                                      ▼
      ===================================================================
      STAGE 1: ROBUST BINARY GATEKEEPER (OK vs. NG Segregation)
      -------------------------------------------------------------------
      Fuses 10-feature vector:
      X_gate = [MOS, NOI, DIS, COL, LOUD, P_OK, P_Cut, P_Noise, P_Rep, P_Wobble]
      • Vetoes "OK" if NISQA detects Noisiness, Discontinuity, or Coloration.
      • Vetoes "OK" if 1D CNN / Self-Similarity detects clean Repetition.
      ===================================================================
                    │                                   │
         [ PASS: Clean Audio ]               [ FAIL: Defect Detected ]
                    ▼                                   ▼
            ┌───────────────┐        ==================================
            │  FINAL: "OK"  │        STAGE 2: DEFECT SUB-CLASSIFIER
            └───────────────┘        ----------------------------------
                                     Zeroes out P_OK and resolves class:
                                     • NG_Noise  (CNN P_Noise + NISQA noi_pred)
                                     • NG_Cut    (CNN P_Cut   + NISQA dis_pred)
                                     • NG_Wobble (CNN P_Wobble+ NISQA col_pred)
                                     • NG_Rep    (CNN P_Rep   + Self-Similarity)
                                     + Extracts 25ms Peak Event Timestamp (s)
                                     ==================================
```

### How This Solves the 68% OK Precision ("Softmax Squeeze") Bottleneck
Previously, using a flat `OK_THRESHOLD = 0.35` on the 5-way 1D CNN allowed ~23% of subtle `NG_Wobble` and `NG_Rep` recordings (scoring `P_OK = 0.36`, `P_Wobble = 0.48`) to leak into the `OK` column.
- **When a normal expressive voice has slight natural fluctuation** (`P_Wobble = 0.38, P_OK = 0.35`), NISQA scores `col_pred > 4.0` and `mos_pred > 4.0`, confirming the file is genuinely **OK**.
- **When an actual subtle wobble occurs** (`P_OK = 0.36`), NISQA's `col_pred` and `mos_pred` drop sharply (`< 3.5`), immediately vetoing `OK` and routing the file to **Stage 2 (`NG_Wobble`)**.
- **When a clean TTS phrase repetition occurs** (`"Turning on... Turning on air conditioning"`), where NISQA scores `> 4.0` because there is no acoustic click, the 1D Dilated CNN (`P_Rep`) + Acoustic Self-Similarity check vetoes `OK` and routes the file to **Stage 2 (`NG_Rep`)**.

---

## 3. Crucial Fix for "Up to 48 kHz" Audio Without Resampling Wobbles

1. **Immediate Zero-Retraining Fix (`--sr 44100`)**:
   In `inference.py`, `load_audio_clean()` uses `librosa.resample(..., res_type='soxr_vhq')` (Very High Quality Soxr sinc interpolation) for the 1D CNN path to match existing 44.1 kHz `scaler.pkl` and `audio_classifier_model.keras` frame sizes (`FRAME_LEN = 2205`, `HOP_LEN = 1102`) with zero phase warping, while passing the untouched native 48 kHz waveform directly into NISQA.
2. **Native 48 kHz Upgrade (`--sr 48000`)**:
   Run `python train_hybrid.py --data_dir ./dataset --sr 48000 --mode full` to train natively at 48 kHz (`FRAME_LEN = 2400` for 50ms, `HOP_LEN = 1200` for 25ms, `n_fft = 2400`) with zero resampling across the entire pipeline.

---

## 4. Project File Inventory & Quick Start (100% Offline)

| File | Purpose |
| :--- | :--- |
| `train_hybrid.py` | Full offline training script for 63-feature `StandardScaler`, 1D Temporal Dilated CNN + `AttentionLayer`, and Stage 1 10-feature `XGBoost` / `Logistic` Meta-Gate. |
| `inference.py` | Production CLI & batch inference engine with Two-Stage Hybrid Gatekeeper, `get_activity_heatmap()` 25ms timestamp extractor, and color-coded Excel `.xlsx` report exporter. |
| `gui_app.py` | Native offline Python Desktop GUI (`tkinter`/`ttk`) for interactive single-file/folder batch inspection, threshold tuning, and Excel export. |
| `requirements.txt` | Pinned Python dependencies for offline wheelhouse installation (`pip install --no-index --find-links=...`). |
| `PREREQUISITES_INSTALLATION.txt` | Complete hardware, OS, and two-phase offline wheelhouse installation guide. |
| `USER_MANUAL.txt` | Detailed operator and audio QA engineer manual covering GUI operation, Stage 1/2 thresholds, and Excel report interpretation. |

### Quick Offline Commands

```bash
# 1. Install dependencies 100% offline from local wheelhouse
pip install --no-index --find-links=offline_package/wheels -r requirements.txt

# 2. Run built-in offline verification self-test (verifies Stage 1 & Stage 2 gate logic)
python inference.py --self-test

# 3. Launch the 100% Offline Desktop GUI Application
python gui_app.py

# 4. Run batch CLI inference on a folder of 48 kHz audio files and generate Excel report
python inference.py --input ./test_audio_folder --output_excel Audio_QA_Hybrid_Report.xlsx --sr 44100 --gate_mode rule_consensus

# 5. Train native 48 kHz 1D CNN + Stage 1 XGBoost Meta-Gate on labeled dataset
python train_hybrid.py --data_dir ./dataset --sr 48000 --epochs 60 --mode full
```
