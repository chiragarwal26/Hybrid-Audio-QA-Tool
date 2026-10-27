import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildXGateVector,
  evaluateHybridGate,
  extractPeakAttentionTimestamp,
  DEFAULT_THRESHOLDS,
} from './hybridPipelineLogic';

test('buildXGateVector fuses 5 NISQA scores and 5 CNN probabilities into 10-element vector', () => {
  const vector = buildXGateVector(
    { mos_pred: 4.35, noi_pred: 4.4, dis_pred: 4.5, col_pred: 4.2, loud_pred: 4.3 },
    { p_ok: 0.72, p_cut: 0.05, p_noise: 0.08, p_rep: 0.05, p_wobble: 0.10 }
  );
  assert.equal(vector.length, 10);
  assert.equal(vector[0], 4.35);
  assert.equal(vector[5], 0.72);
  assert.equal(vector[9], 0.10);
});

test('Stage 1 fixes the 68% OK precision bottleneck: normal speech with Softmax Squeeze (P_OK=0.35, P_Wobble=0.38) passes as OK when NISQA col_pred & mos_pred > 4.0', () => {
  const result = evaluateHybridGate(
    { mos_pred: 4.25, noi_pred: 4.3, dis_pred: 4.4, col_pred: 4.15, loud_pred: 4.2 },
    { p_ok: 0.35, p_cut: 0.08, p_noise: 0.09, p_rep: 0.10, p_wobble: 0.38 },
    DEFAULT_THRESHOLDS
  );
  assert.equal(result.stage1Status, 'OK');
  assert.equal(result.finalClass, 'OK');
  assert.equal(result.vetoReasons.length, 0);
});

test('Stage 1 catches subtle NG_Wobble leak: when P_OK=0.36 (which previously leaked to OK) has low NISQA col_pred=2.4, NISQA vetoes OK and routes to NG_Wobble', () => {
  const result = evaluateHybridGate(
    { mos_pred: 2.75, noi_pred: 4.1, dis_pred: 4.0, col_pred: 2.35, loud_pred: 3.9 },
    { p_ok: 0.36, p_cut: 0.05, p_noise: 0.06, p_rep: 0.05, p_wobble: 0.48 },
    DEFAULT_THRESHOLDS
  );
  assert.equal(result.stage1Status, 'NG');
  assert.equal(result.finalClass, 'NG_Wobble');
  assert.ok(result.vetoReasons.length >= 1);
});

test('Stage 1 catches clean TTS phrase repetition (NG_Rep) where NISQA scores > 4.0 but 1D CNN P_Rep = 0.68', () => {
  const result = evaluateHybridGate(
    { mos_pred: 4.3, noi_pred: 4.4, dis_pred: 4.1, col_pred: 4.3, loud_pred: 4.2 },
    { p_ok: 0.22, p_cut: 0.04, p_noise: 0.03, p_rep: 0.68, p_wobble: 0.03 },
    DEFAULT_THRESHOLDS
  );
  assert.equal(result.stage1Status, 'NG');
  assert.equal(result.finalClass, 'NG_Rep');
});

test('extractPeakAttentionTimestamp pinpoints 25ms frame index accurately', () => {
  const weights = [0.05, 0.1, 0.12, 0.94, 0.2, 0.08];
  const peak = extractPeakAttentionTimestamp(weights, 0.025);
  assert.equal(peak.peakFrameIndex, 3);
  assert.equal(peak.peakTimestampSec, 0.075);
  assert.equal(peak.peakWeight, 0.94);
});
