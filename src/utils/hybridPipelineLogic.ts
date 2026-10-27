/**
 * Core mathematical and rule-gate logic for the Two-Stage Cascaded Hybrid Audio QA Pipeline
 * Combines NISQA v2.0 (5 perceptual dimensions) + 1D Temporal Dilated CNN + Attention (5 softmax probabilities)
 */

export interface NisqaScores {
  mos_pred: number;   // 1.0 - 5.0 Overall Quality
  noi_pred: number;   // 1.0 - 5.0 Noisiness (1 = very noisy, 5 = clean)
  dis_pred: number;   // 1.0 - 5.0 Discontinuity (1 = severe cuts/drops, 5 = continuous)
  col_pred: number;   // 1.0 - 5.0 Coloration (1 = severe phase/freq wobble, 5 = natural)
  loud_pred: number;  // 1.0 - 5.0 Loudness (1 = abnormal gain drop, 5 = optimal level)
}

export interface CnnProbabilities {
  p_ok: number;
  p_cut: number;
  p_noise: number;
  p_rep: number;
  p_wobble: number;
}

export interface GateThresholds {
  mosMin: number;       // default 3.6
  noiMin: number;       // default 3.5
  disMin: number;       // default 3.5
  colMin: number;       // default 3.5
  pOkMin: number;       // default 0.35
  maxDefectCap: number; // default 0.55 for max(P_Rep, P_Cut, P_Noise)
  gateMode: 'rule_consensus' | 'xgb_meta';
}

export type DefectClass = 'OK' | 'NG_Noise' | 'NG_Cut' | 'NG_Wobble' | 'NG_Rep';

export interface HybridGateResult {
  xGateVector: number[]; // 10-element fused feature vector
  stage1Status: 'OK' | 'NG';
  nisqaPass: boolean;
  cnnTemporalPass: boolean;
  vetoReasons: string[];
  finalClass: DefectClass;
  confidenceScore: number;
  stage2DiagnosticScores: {
    NG_Noise: number;
    NG_Cut: number;
    NG_Wobble: number;
    NG_Rep: number;
  };
}

export const DEFAULT_THRESHOLDS: GateThresholds = {
  mosMin: 3.6,
  noiMin: 3.5,
  disMin: 3.5,
  colMin: 3.5,
  pOkMin: 0.35,
  maxDefectCap: 0.55,
  gateMode: 'rule_consensus',
};

/**
 * Constructs the 10-element Stage 1 Gatekeeper feature vector:
 * X_gate = [MOS, NOI, DIS, COL, LOUD, P_OK, P_Cut, P_Noise, P_Rep, P_Wobble]
 */
export function buildXGateVector(nisqa: NisqaScores, cnn: CnnProbabilities): number[] {
  return [
    Number(nisqa.mos_pred.toFixed(4)),
    Number(nisqa.noi_pred.toFixed(4)),
    Number(nisqa.dis_pred.toFixed(4)),
    Number(nisqa.col_pred.toFixed(4)),
    Number(nisqa.loud_pred.toFixed(4)),
    Number(cnn.p_ok.toFixed(4)),
    Number(cnn.p_cut.toFixed(4)),
    Number(cnn.p_noise.toFixed(4)),
    Number(cnn.p_rep.toFixed(4)),
    Number(cnn.p_wobble.toFixed(4)),
  ];
}

/**
 * Evaluates Stage 1 (Robust Binary Gatekeeper: OK vs. NG Segregation)
 * and Stage 2 (NG Defect Sub-Classifier: NG_Noise, NG_Cut, NG_Wobble, NG_Rep).
 */
export function evaluateHybridGate(
  nisqa: NisqaScores,
  cnn: CnnProbabilities,
  thresholds: GateThresholds = DEFAULT_THRESHOLDS
): HybridGateResult {
  const xGateVector = buildXGateVector(nisqa, cnn);
  const vetoReasons: string[] = [];

  // Perceptual Check (NISQA v2.0 native 48 kHz)
  const mosOk = nisqa.mos_pred >= thresholds.mosMin;
  const noiOk = nisqa.noi_pred >= thresholds.noiMin;
  const disOk = nisqa.dis_pred >= thresholds.disMin;
  const colOk = nisqa.col_pred >= thresholds.colMin;
  const nisqaPass = mosOk && noiOk && disOk && colOk;

  if (!mosOk) vetoReasons.push(`NISQA MOS (${nisqa.mos_pred.toFixed(2)}) < ${thresholds.mosMin.toFixed(2)}`);
  if (!noiOk) vetoReasons.push(`NISQA Noisiness (${nisqa.noi_pred.toFixed(2)}) < ${thresholds.noiMin.toFixed(2)}`);
  if (!disOk) vetoReasons.push(`NISQA Discontinuity (${nisqa.dis_pred.toFixed(2)}) < ${thresholds.disMin.toFixed(2)}`);
  if (!colOk) vetoReasons.push(`NISQA Coloration (${nisqa.col_pred.toFixed(2)}) < ${thresholds.colMin.toFixed(2)}`);

  // Temporal & Defect Check (1D Dilated CNN + Self-Similarity)
  const maxHardDefect = Math.max(cnn.p_rep, cnn.p_cut, cnn.p_noise);
  const pOkPass = cnn.p_ok >= thresholds.pOkMin;
  const defectCapPass = maxHardDefect < thresholds.maxDefectCap;
  const cnnTemporalPass = pOkPass && defectCapPass;

  if (!pOkPass) {
    vetoReasons.push(`1D CNN P_OK (${cnn.p_ok.toFixed(2)}) < ${thresholds.pOkMin.toFixed(2)}`);
  }
  if (!defectCapPass) {
    vetoReasons.push(`1D CNN max(P_Rep, P_Cut, P_Noise) (${maxHardDefect.toFixed(2)}) >= ${thresholds.maxDefectCap.toFixed(2)}`);
  }

  let stage1Status: 'OK' | 'NG';

  if (thresholds.gateMode === 'rule_consensus') {
    stage1Status = nisqaPass && cnnTemporalPass ? 'OK' : 'NG';
  } else {
    // Calibrated Logistic / XGBoost Meta-Classifier scoring on the 10-feature vector
    // Normalizes NISQA scores from [1, 5] to [0, 1] penalty and combines with CNN probabilities
    const nisqaPenalty =
      Math.max(0, (4.2 - nisqa.mos_pred) * 0.45) +
      Math.max(0, (4.0 - nisqa.noi_pred) * 0.35) +
      Math.max(0, (4.0 - nisqa.dis_pred) * 0.38) +
      Math.max(0, (4.0 - nisqa.col_pred) * 0.35);

    const cnnDefectSignal =
      cnn.p_rep * 1.35 +
      cnn.p_cut * 1.15 +
      cnn.p_noise * 1.1 +
      cnn.p_wobble * 0.85 -
      cnn.p_ok * 0.95;

    const logit = -0.85 + nisqaPenalty + cnnDefectSignal;
    const probNg = 1 / (1 + Math.exp(-3.2 * logit));
    stage1Status = probNg >= 0.5 ? 'NG' : 'OK';
    if (stage1Status === 'NG' && vetoReasons.length === 0) {
      vetoReasons.push(`Meta-Classifier P(NG) = ${(probNg * 100).toFixed(1)}% >= 50.0%`);
    }
  }

  // Stage 2: Defect Sub-Classifier (Zeroes out P_OK and resolves exact NG defect using NISQA dimensions)
  // Severity deficit on [0, 1] scale where 1.0 = worst impairment (score = 1.0) and 0.0 = pristine (score = 5.0)
  const noiDeficit = Math.max(0, Math.min(1, (5.0 - nisqa.noi_pred) / 4.0));
  const disDeficit = Math.max(0, Math.min(1, (5.0 - nisqa.dis_pred) / 4.0));
  const colDeficit = Math.max(0, Math.min(1, (5.0 - nisqa.col_pred) / 4.0));

  // Clean acoustic bonus when NISQA sees pristine acoustics (isolating clean TTS/infotainment repetition)
  const acousticCleanliness = Math.max(0, Math.min(1, ((nisqa.noi_pred + nisqa.col_pred) / 2 - 2.5) / 2.5));

  const rawScores = {
    NG_Noise: cnn.p_noise * 0.65 + noiDeficit * 0.35,
    NG_Cut: cnn.p_cut * 0.65 + disDeficit * 0.35,
    NG_Wobble: cnn.p_wobble * 0.60 + colDeficit * 0.40,
    // NG_Rep is boosted when CNN P_Rep is high; even if NISQA dis_pred drops slightly on abrupt splice,
    // or if NISQA noi_pred/col_pred remain clean on smooth phrase repetition
    NG_Rep: cnn.p_rep * 0.78 + acousticCleanliness * cnn.p_rep * 0.22,
  };

  const totalRaw =
    rawScores.NG_Noise + rawScores.NG_Cut + rawScores.NG_Wobble + rawScores.NG_Rep || 1;

  const stage2DiagnosticScores = {
    NG_Noise: Number((rawScores.NG_Noise / totalRaw).toFixed(4)),
    NG_Cut: Number((rawScores.NG_Cut / totalRaw).toFixed(4)),
    NG_Wobble: Number((rawScores.NG_Wobble / totalRaw).toFixed(4)),
    NG_Rep: Number((rawScores.NG_Rep / totalRaw).toFixed(4)),
  };

  if (stage1Status === 'OK') {
    const mosNorm = Math.max(0, Math.min(1, (nisqa.mos_pred - 1) / 4));
    const confidenceScore = Number(((mosNorm * 0.55 + cnn.p_ok * 0.45) * 100).toFixed(1));
    return {
      xGateVector,
      stage1Status: 'OK',
      nisqaPass,
      cnnTemporalPass,
      vetoReasons: [],
      finalClass: 'OK',
      confidenceScore,
      stage2DiagnosticScores,
    };
  }

  // Pick winning NG sub-class
  const entries = Object.entries(stage2DiagnosticScores) as [Exclude<DefectClass, 'OK'>, number][];
  entries.sort((a, b) => b[1] - a[1]);
  const [winningDefect, winningScore] = entries[0];

  return {
    xGateVector,
    stage1Status: 'NG',
    nisqaPass,
    cnnTemporalPass,
    vetoReasons,
    finalClass: winningDefect,
    confidenceScore: Number((winningScore * 100).toFixed(1)),
    stage2DiagnosticScores,
  };
}

/**
 * Extracts the peak defect timestamp (in seconds) from a 25ms-resolution temporal attention array
 */
export function extractPeakAttentionTimestamp(
  attentionWeights: number[],
  hopSeconds: number = 0.025
): { peakTimestampSec: number; peakWeight: number; peakFrameIndex: number } {
  if (!attentionWeights || attentionWeights.length === 0) {
    return { peakTimestampSec: 0, peakWeight: 0, peakFrameIndex: 0 };
  }
  let maxIdx = 0;
  let maxVal = attentionWeights[0];
  for (let i = 1; i < attentionWeights.length; i++) {
    if (attentionWeights[i] > maxVal) {
      maxVal = attentionWeights[i];
      maxIdx = i;
    }
  }
  return {
    peakTimestampSec: Number((maxIdx * hopSeconds).toFixed(3)),
    peakWeight: Number(maxVal.toFixed(4)),
    peakFrameIndex: maxIdx,
  };
}
