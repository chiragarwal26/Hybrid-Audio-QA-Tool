import { NisqaScores, CnnProbabilities } from '../utils/hybridPipelineLogic';

export interface AudioSampleRecord {
  id: string;
  filename: string;
  sampleRateHz: number;
  durationSec: number;
  channels: 1;
  format: 'WAV' | 'FLAC' | 'MP3';
  description: string;
  spokenTranscript: string;
  groundTruth: 'OK' | 'NG_Noise' | 'NG_Cut' | 'NG_Wobble' | 'NG_Rep';
  legacyPipelinePrediction: 'OK' | 'NG_Noise' | 'NG_Cut' | 'NG_Wobble' | 'NG_Rep';
  nisqa: NisqaScores;
  cnn: CnnProbabilities;
  selfSimilarityScore: number; // Acoustic self-similarity matrix off-diagonal peak [0, 1]
  attentionHeatmap: number[];  // 40 frames (each representing 100ms / 25ms aggregated peak window)
  waveformEnvelope: number[];  // 60 points for waveform display
  synthType: 'clean' | 'noise' | 'cut' | 'wobble' | 'rep' | 'borderline_ok';
}

export const SAMPLE_AUDIO_DATASET: AudioSampleRecord[] = [
  {
    id: 'AUD-48K-001',
    filename: 'infotainment_nav_prompt_01.wav',
    sampleRateHz: 48000,
    durationSec: 4.0,
    channels: 1,
    format: 'WAV',
    description: 'Pristine 48 kHz Fullband navigation voice prompt with natural prosody.',
    spokenTranscript: '"In three hundred meters, keep left towards the highway interchange."',
    groundTruth: 'OK',
    legacyPipelinePrediction: 'OK',
    nisqa: {
      mos_pred: 4.52,
      noi_pred: 4.65,
      dis_pred: 4.70,
      col_pred: 4.48,
      loud_pred: 4.40,
    },
    cnn: {
      p_ok: 0.86,
      p_cut: 0.02,
      p_noise: 0.03,
      p_rep: 0.03,
      p_wobble: 0.06,
    },
    selfSimilarityScore: 0.14,
    attentionHeatmap: [
      0.08, 0.09, 0.11, 0.12, 0.10, 0.13, 0.12, 0.11, 0.09, 0.10,
      0.12, 0.14, 0.11, 0.10, 0.09, 0.11, 0.13, 0.12, 0.10, 0.08,
      0.09, 0.11, 0.12, 0.10, 0.09, 0.11, 0.10, 0.08, 0.07, 0.06
    ],
    waveformEnvelope: [
      0.15, 0.42, 0.68, 0.74, 0.55, 0.30, 0.62, 0.81, 0.76, 0.49,
      0.22, 0.58, 0.79, 0.85, 0.64, 0.41, 0.72, 0.69, 0.53, 0.28,
      0.61, 0.77, 0.82, 0.59, 0.35, 0.66, 0.71, 0.48, 0.25, 0.12
    ],
    synthType: 'clean',
  },
  {
    id: 'AUD-48K-002',
    filename: 'hvac_command_clean_rep_04.wav',
    sampleRateHz: 48000,
    durationSec: 4.2,
    channels: 1,
    format: 'WAV',
    description: 'Clean TTS buffer stutter ("Turning on... Turning on air conditioning"). NISQA scores > 4.1 acoustically, but 1D CNN + Self-Similarity catches the phrase loop.',
    spokenTranscript: '"Turning on... Turning on the cabin air conditioning system."',
    groundTruth: 'NG_Rep',
    legacyPipelinePrediction: 'OK', // Previously leaked in Fold 5 when P_OK hit 0.36
    nisqa: {
      mos_pred: 4.18,
      noi_pred: 4.45,
      dis_pred: 3.92,
      col_pred: 4.36,
      loud_pred: 4.25,
    },
    cnn: {
      p_ok: 0.24,
      p_cut: 0.05,
      p_noise: 0.02,
      p_rep: 0.65,
      p_wobble: 0.04,
    },
    selfSimilarityScore: 0.89,
    attentionHeatmap: [
      0.10, 0.12, 0.15, 0.18, 0.22, 0.19, 0.14, 0.12, 0.45, 0.78,
      0.94, 0.88, 0.62, 0.30, 0.18, 0.15, 0.14, 0.12, 0.11, 0.10,
      0.12, 0.11, 0.09, 0.08, 0.09, 0.10, 0.08, 0.07, 0.06, 0.05
    ],
    waveformEnvelope: [
      0.20, 0.65, 0.82, 0.71, 0.48, 0.15, 0.20, 0.65, 0.82, 0.71,
      0.48, 0.32, 0.74, 0.88, 0.79, 0.61, 0.45, 0.69, 0.75, 0.52,
      0.38, 0.62, 0.70, 0.58, 0.41, 0.55, 0.49, 0.33, 0.19, 0.08
    ],
    synthType: 'rep',
  },
  {
    id: 'AUD-48K-003',
    filename: 'bt_handsfree_clock_drift_09.flac',
    sampleRateHz: 48000,
    durationSec: 3.8,
    channels: 1,
    format: 'FLAC',
    description: 'Subtle fullband clock-recovery pitch modulation (NG_Wobble). Previously leaked into OK because P_OK = 0.36 >= 0.35. NISQA col_pred (2.28) immediately vetoes OK.',
    spokenTranscript: '"Connecting to primary mobile device via Bluetooth audio profile."',
    groundTruth: 'NG_Wobble',
    legacyPipelinePrediction: 'OK', // Classic Fold 5 Softmax Squeeze leak fixed by Hybrid Gate
    nisqa: {
      mos_pred: 2.68,
      noi_pred: 4.12,
      dis_pred: 3.95,
      col_pred: 2.28,
      loud_pred: 3.88,
    },
    cnn: {
      p_ok: 0.36,
      p_cut: 0.04,
      p_noise: 0.05,
      p_rep: 0.06,
      p_wobble: 0.49,
    },
    selfSimilarityScore: 0.21,
    attentionHeatmap: [
      0.12, 0.14, 0.19, 0.28, 0.35, 0.42, 0.58, 0.76, 0.89, 0.91,
      0.84, 0.72, 0.61, 0.48, 0.39, 0.33, 0.29, 0.25, 0.22, 0.18,
      0.16, 0.15, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06
    ],
    waveformEnvelope: [
      0.18, 0.52, 0.74, 0.68, 0.59, 0.63, 0.77, 0.84, 0.69, 0.54,
      0.61, 0.79, 0.83, 0.66, 0.51, 0.58, 0.73, 0.67, 0.49, 0.42,
      0.56, 0.68, 0.61, 0.47, 0.39, 0.51, 0.44, 0.31, 0.18, 0.09
    ],
    synthType: 'wobble',
  },
  {
    id: 'AUD-48K-004',
    filename: 'mic_array_hf_hiss_14khz.wav',
    sampleRateHz: 48000,
    durationSec: 3.5,
    channels: 1,
    format: 'WAV',
    description: '14 kHz high-frequency EMI static hiss on voice capture (invisible to 8 kHz P.563, caught natively by 48 kHz NISQA noi_pred = 2.14 + 1D CNN P_Noise = 0.71).',
    spokenTranscript: '"Setting cabin temperature to twenty-two degrees Celsius."',
    groundTruth: 'NG_Noise',
    legacyPipelinePrediction: 'NG_Noise',
    nisqa: {
      mos_pred: 2.45,
      noi_pred: 2.14,
      dis_pred: 4.10,
      col_pred: 3.65,
      loud_pred: 3.92,
    },
    cnn: {
      p_ok: 0.11,
      p_cut: 0.06,
      p_noise: 0.71,
      p_rep: 0.04,
      p_wobble: 0.08,
    },
    selfSimilarityScore: 0.18,
    attentionHeatmap: [
      0.45, 0.52, 0.61, 0.68, 0.74, 0.79, 0.83, 0.86, 0.82, 0.78,
      0.75, 0.79, 0.84, 0.88, 0.81, 0.76, 0.72, 0.69, 0.65, 0.62,
      0.58, 0.55, 0.52, 0.49, 0.46, 0.44, 0.41, 0.38, 0.35, 0.32
    ],
    waveformEnvelope: [
      0.35, 0.68, 0.85, 0.79, 0.62, 0.48, 0.74, 0.89, 0.83, 0.67,
      0.51, 0.76, 0.91, 0.84, 0.69, 0.55, 0.78, 0.74, 0.63, 0.49,
      0.69, 0.81, 0.77, 0.64, 0.52, 0.68, 0.63, 0.51, 0.42, 0.36
    ],
    synthType: 'noise',
  },
  {
    id: 'AUD-48K-005',
    filename: 'dma_buffer_underrun_cut_12.wav',
    sampleRateHz: 44100,
    durationSec: 3.6,
    channels: 1,
    format: 'WAV',
    description: '44.1 kHz capture with a 65ms DMA dropout / zero-fill splice at 1.45s. Soxr_vhq resampler prevents artificial pitch wobble; NISQA dis_pred = 1.95 & CNN P_Cut = 0.78.',
    spokenTranscript: '"Warning: lane departure... [65ms drop] ...system calibration required."',
    groundTruth: 'NG_Cut',
    legacyPipelinePrediction: 'NG_Cut',
    nisqa: {
      mos_pred: 2.32,
      noi_pred: 4.20,
      dis_pred: 1.95,
      col_pred: 3.82,
      loud_pred: 3.45,
    },
    cnn: {
      p_ok: 0.08,
      p_cut: 0.78,
      p_noise: 0.05,
      p_rep: 0.05,
      p_wobble: 0.04,
    },
    selfSimilarityScore: 0.16,
    attentionHeatmap: [
      0.07, 0.08, 0.09, 0.11, 0.10, 0.12, 0.11, 0.14, 0.18, 0.25,
      0.64, 0.97, 0.89, 0.42, 0.16, 0.12, 0.10, 0.09, 0.11, 0.10,
      0.08, 0.09, 0.07, 0.08, 0.06, 0.07, 0.06, 0.05, 0.05, 0.04
    ],
    waveformEnvelope: [
      0.22, 0.64, 0.81, 0.75, 0.59, 0.72, 0.84, 0.79, 0.68, 0.55,
      0.02, 0.00, 0.01, 0.76, 0.88, 0.82, 0.67, 0.53, 0.71, 0.78,
      0.64, 0.48, 0.62, 0.73, 0.58, 0.42, 0.51, 0.39, 0.21, 0.10
    ],
    synthType: 'cut',
  },
  {
    id: 'AUD-48K-006',
    filename: 'expressive_voice_vibrato_ok_18.wav',
    sampleRateHz: 44100,
    durationSec: 4.1,
    channels: 1,
    format: 'WAV',
    description: 'Expressive human speaker with natural vocal micro-tremor (P_Wobble = 0.38, P_OK = 0.35). NISQA confirms high col_pred (4.22) and mos_pred (4.31), preventing false NG.',
    spokenTranscript: '"Welcome aboard! Your destination route has been updated with live traffic."',
    groundTruth: 'OK',
    legacyPipelinePrediction: 'NG_Wobble', // False alarm without threshold hack; Hybrid Gate resolves cleanly
    nisqa: {
      mos_pred: 4.31,
      noi_pred: 4.48,
      dis_pred: 4.55,
      col_pred: 4.22,
      loud_pred: 4.35,
    },
    cnn: {
      p_ok: 0.35,
      p_cut: 0.07,
      p_noise: 0.09,
      p_rep: 0.11,
      p_wobble: 0.38,
    },
    selfSimilarityScore: 0.19,
    attentionHeatmap: [
      0.11, 0.13, 0.15, 0.18, 0.21, 0.24, 0.22, 0.19, 0.17, 0.16,
      0.18, 0.22, 0.25, 0.21, 0.18, 0.15, 0.14, 0.16, 0.19, 0.17,
      0.15, 0.13, 0.12, 0.14, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07
    ],
    waveformEnvelope: [
      0.19, 0.58, 0.79, 0.85, 0.67, 0.45, 0.69, 0.82, 0.74, 0.56,
      0.38, 0.64, 0.81, 0.87, 0.71, 0.52, 0.76, 0.80, 0.65, 0.43,
      0.61, 0.75, 0.79, 0.62, 0.44, 0.68, 0.72, 0.51, 0.29, 0.14
    ],
    synthType: 'borderline_ok',
  },
];
