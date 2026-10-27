import React, { useState, useMemo, useRef } from 'react';
import {
  Play,
  Square,
  Upload,
  Download,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Sliders,
} from 'lucide-react';
import { SAMPLE_AUDIO_DATASET, AudioSampleRecord } from '../data/sampleAudioDataset';
import {
  DEFAULT_THRESHOLDS,
  GateThresholds,
  evaluateHybridGate,
  extractPeakAttentionTimestamp,
} from '../utils/hybridPipelineLogic';
import {
  playSynthesizedSample,
  stopActiveAudio,
  analyzeUploadedAudioFile,
} from '../utils/audioSynth';

export const InteractiveWorkbench: React.FC = () => {
  const [samples, setSamples] = useState<AudioSampleRecord[]>(SAMPLE_AUDIO_DATASET);
  const [selectedId, setSelectedId] = useState<string>(SAMPLE_AUDIO_DATASET[2].id); // Default to the subtle NG_Wobble leak fix case
  const [thresholds, setThresholds] = useState<GateThresholds>(DEFAULT_THRESHOLDS);
  const [srMode, setSrMode] = useState<44100 | 48000>(48000);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [filterVerdict, setFilterVerdict] = useState<'ALL' | 'OK' | 'NG'>('ALL');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Evaluate all samples dynamically against the current Stage 1 & Stage 2 thresholds
  const evaluatedRecords = useMemo(() => {
    return samples.map((sample) => {
      const gate = evaluateHybridGate(sample.nisqa, sample.cnn, thresholds);
      const hopSec = Number((sample.durationSec / sample.attentionHeatmap.length).toFixed(3));
      const peakInfo = extractPeakAttentionTimestamp(sample.attentionHeatmap, hopSec);
      return {
        sample,
        gate,
        peakTimestampSec: gate.stage1Status === 'OK' ? 0.0 : peakInfo.peakTimestampSec,
        peakFrameIndex: peakInfo.peakFrameIndex,
      };
    });
  }, [samples, thresholds]);

  const filteredRecords = useMemo(() => {
    if (filterVerdict === 'ALL') return evaluatedRecords;
    return evaluatedRecords.filter((r) => r.gate.stage1Status === filterVerdict);
  }, [evaluatedRecords, filterVerdict]);

  const activeEntry =
    evaluatedRecords.find((r) => r.sample.id === selectedId) || evaluatedRecords[0];

  // Batch summary statistics
  const stats = useMemo(() => {
    const total = evaluatedRecords.length;
    const okCount = evaluatedRecords.filter((r) => r.gate.stage1Status === 'OK').length;
    const ngCount = total - okCount;
    const exactMatchCount = evaluatedRecords.filter(
      (r) => r.gate.finalClass === r.sample.groundTruth
    ).length;
    const legacyExactCount = evaluatedRecords.filter(
      (r) => r.sample.legacyPipelinePrediction === r.sample.groundTruth
    ).length;
    return {
      total,
      okCount,
      ngCount,
      accuracyPct: total > 0 ? Math.round((exactMatchCount / total) * 100) : 100,
      legacyAccuracyPct: total > 0 ? Math.round((legacyExactCount / total) * 100) : 67,
    };
  }, [evaluatedRecords]);

  const handleTogglePlay = (sample: AudioSampleRecord) => {
    if (playingId === sample.id) {
      stopActiveAudio();
      setPlayingId(null);
    } else {
      setPlayingId(sample.id);
      playSynthesizedSample(sample, () => {
        setPlayingId(null);
      });
    }
  };

  const handleUploadChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const record = await analyzeUploadedAudioFile(file);
      setSamples((prev) => [record, ...prev]);
      setSelectedId(record.id);
    } catch {
      // ignore invalid audio format
    } finally {
      e.target.value = '';
    }
  };

  const handleExportCsvReport = () => {
    const headers = [
      'ID',
      'Filename',
      'Sample_Rate_Hz',
      'Duration_Sec',
      'Stage1_Binary_Verdict',
      'Stage2_Final_Class',
      'Ground_Truth',
      'Event_Timestamp_Sec',
      'NISQA_MOS',
      'NISQA_NOI',
      'NISQA_DIS',
      'NISQA_COL',
      'NISQA_LOUD',
      'CNN_P_OK',
      'CNN_P_Cut',
      'CNN_P_Noise',
      'CNN_P_Rep',
      'CNN_P_Wobble',
      'Self_Similarity',
      'Gate_Veto_Reasons',
    ];
    const rows = evaluatedRecords.map(({ sample, gate, peakTimestampSec }) => [
      sample.id,
      `"${sample.filename}"`,
      sample.sampleRateHz,
      sample.durationSec.toFixed(2),
      gate.stage1Status,
      gate.finalClass,
      sample.groundTruth,
      peakTimestampSec.toFixed(3),
      sample.nisqa.mos_pred.toFixed(2),
      sample.nisqa.noi_pred.toFixed(2),
      sample.nisqa.dis_pred.toFixed(2),
      sample.nisqa.col_pred.toFixed(2),
      sample.nisqa.loud_pred.toFixed(2),
      sample.cnn.p_ok.toFixed(3),
      sample.cnn.p_cut.toFixed(3),
      sample.cnn.p_noise.toFixed(3),
      sample.cnn.p_rep.toFixed(3),
      sample.cnn.p_wobble.toFixed(3),
      sample.selfSimilarityScore.toFixed(3),
      `"${gate.vetoReasons.join('; ') || 'Passed All Checks'}"`,
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Audio_QA_Hybrid_Report.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Full-Width 4-Column KPI Summary Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="text-xs text-slate-400">Batch Files Inspected (Up to 48 kHz)</div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono tabular-nums text-slate-100">
              {stats.total}
            </span>
            <span className="text-xs text-slate-400 font-mono tabular-nums">
              OK: {stats.okCount} · NG: {stats.ngCount}
            </span>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="text-xs text-slate-400">Stage 1 + Stage 2 Hybrid Gate Accuracy</div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono tabular-nums text-emerald-400">
              {stats.accuracyPct}%
            </span>
            <span className="text-xs text-slate-400 font-mono tabular-nums">
              Legacy CNN: {stats.legacyAccuracyPct}%
            </span>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="text-xs text-slate-400">1D CNN Resampling Pipeline Mode</div>
          <div className="mt-1.5 flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setSrMode(48000)}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors cursor-pointer ${
                srMode === 48000
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              48 kHz Native
            </button>
            <button
              type="button"
              onClick={() => setSrMode(44100)}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors cursor-pointer ${
                srMode === 44100
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              44.1 kHz (soxr_vhq)
            </button>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="text-xs text-slate-400">Stage 1 Binary Gatekeeper Engine</div>
          <div className="mt-1.5 flex items-center gap-1.5">
            <button
              type="button"
              onClick={() =>
                setThresholds((prev) => ({ ...prev, gateMode: 'rule_consensus' }))
              }
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors cursor-pointer ${
                thresholds.gateMode === 'rule_consensus'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Rule Consensus
            </button>
            <button
              type="button"
              onClick={() =>
                setThresholds((prev) => ({ ...prev, gateMode: 'xgb_meta' }))
              }
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors cursor-pointer ${
                thresholds.gateMode === 'xgb_meta'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              XGBoost Meta-Gate
            </button>
          </div>
        </div>
      </div>

      {/* Stage 1 Consensus Gate Threshold Tuner Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-sky-400" />
            <h2 className="text-sm font-semibold text-slate-100">
              Stage 1 Binary Gatekeeper Threshold Controls (Live Calibration)
            </h2>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setThresholds(DEFAULT_THRESHOLDS)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Recommended Defaults
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.flac,.mp3,audio/*"
              onChange={handleUploadChange}
              className="hidden"
              aria-label="Upload custom audio file"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5 text-sky-400" />
              Inspect Local Audio (.wav/.flac/.mp3)
            </button>
            <button
              type="button"
              onClick={handleExportCsvReport}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-sky-600 hover:bg-sky-500 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Export QA Report (.csv)
            </button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <div>
            <label
              htmlFor="slider-mos"
              className="flex items-center justify-between text-xs text-slate-300 font-mono"
            >
              <span>NISQA MOS &gt;=</span>
              <span className="text-sky-400 font-semibold tabular-nums">
                {thresholds.mosMin.toFixed(2)}
              </span>
            </label>
            <input
              id="slider-mos"
              type="range"
              min="2.50"
              max="4.50"
              step="0.05"
              value={thresholds.mosMin}
              onChange={(e) =>
                setThresholds((p) => ({ ...p, mosMin: parseFloat(e.target.value) }))
              }
              className="w-full mt-1.5 accent-sky-500 cursor-pointer"
            />
          </div>

          <div>
            <label
              htmlFor="slider-noi"
              className="flex items-center justify-between text-xs text-slate-300 font-mono"
            >
              <span>NISQA NOI &gt;=</span>
              <span className="text-sky-400 font-semibold tabular-nums">
                {thresholds.noiMin.toFixed(2)}
              </span>
            </label>
            <input
              id="slider-noi"
              type="range"
              min="2.00"
              max="4.50"
              step="0.05"
              value={thresholds.noiMin}
              onChange={(e) =>
                setThresholds((p) => ({ ...p, noiMin: parseFloat(e.target.value) }))
              }
              className="w-full mt-1.5 accent-sky-500 cursor-pointer"
            />
          </div>

          <div>
            <label
              htmlFor="slider-dis"
              className="flex items-center justify-between text-xs text-slate-300 font-mono"
            >
              <span>NISQA DIS &gt;=</span>
              <span className="text-sky-400 font-semibold tabular-nums">
                {thresholds.disMin.toFixed(2)}
              </span>
            </label>
            <input
              id="slider-dis"
              type="range"
              min="2.00"
              max="4.50"
              step="0.05"
              value={thresholds.disMin}
              onChange={(e) =>
                setThresholds((p) => ({ ...p, disMin: parseFloat(e.target.value) }))
              }
              className="w-full mt-1.5 accent-sky-500 cursor-pointer"
            />
          </div>

          <div>
            <label
              htmlFor="slider-col"
              className="flex items-center justify-between text-xs text-slate-300 font-mono"
            >
              <span>NISQA COL &gt;=</span>
              <span className="text-sky-400 font-semibold tabular-nums">
                {thresholds.colMin.toFixed(2)}
              </span>
            </label>
            <input
              id="slider-col"
              type="range"
              min="2.00"
              max="4.50"
              step="0.05"
              value={thresholds.colMin}
              onChange={(e) =>
                setThresholds((p) => ({ ...p, colMin: parseFloat(e.target.value) }))
              }
              className="w-full mt-1.5 accent-sky-500 cursor-pointer"
            />
          </div>

          <div>
            <label
              htmlFor="slider-pok"
              className="flex items-center justify-between text-xs text-slate-300 font-mono"
            >
              <span>CNN P_OK &gt;=</span>
              <span className="text-sky-400 font-semibold tabular-nums">
                {thresholds.pOkMin.toFixed(2)}
              </span>
            </label>
            <input
              id="slider-pok"
              type="range"
              min="0.10"
              max="0.75"
              step="0.05"
              value={thresholds.pOkMin}
              onChange={(e) =>
                setThresholds((p) => ({ ...p, pOkMin: parseFloat(e.target.value) }))
              }
              className="w-full mt-1.5 accent-sky-500 cursor-pointer"
            />
          </div>

          <div>
            <label
              htmlFor="slider-defcap"
              className="flex items-center justify-between text-xs text-slate-300 font-mono"
            >
              <span>Max Defect &lt;</span>
              <span className="text-sky-400 font-semibold tabular-nums">
                {thresholds.maxDefectCap.toFixed(2)}
              </span>
            </label>
            <input
              id="slider-defcap"
              type="range"
              min="0.25"
              max="0.85"
              step="0.05"
              value={thresholds.maxDefectCap}
              onChange={(e) =>
                setThresholds((p) => ({
                  ...p,
                  maxDefectCap: parseFloat(e.target.value),
                }))
              }
              className="w-full mt-1.5 accent-sky-500 cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Main Split: Left Batch Table (7 cols) + Right Deep Diagnostic Inspector (5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left: Batch Audio QA Table */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-lg overflow-hidden flex flex-col justify-between">
          <div>
            <div className="p-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">
                  Batch Audio QA Inspection Queue (Click Any File to Inspect)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Includes Fold 5 Softmax Squeeze edge cases (NG_Wobble &amp; NG_Rep leaks vs. expressive OK speech).
                </p>
              </div>
              <div className="flex items-center gap-1 p-1 bg-slate-950 rounded-lg border border-slate-800">
                {(['ALL', 'OK', 'NG'] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setFilterVerdict(mode)}
                    className={`px-2.5 py-1 text-xs font-medium rounded transition-colors cursor-pointer ${
                      filterVerdict === mode
                        ? 'bg-slate-800 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-950/60 text-[11px] font-medium text-slate-400">
                    <th className="py-2.5 px-3">Play</th>
                    <th className="py-2.5 px-3">Audio File</th>
                    <th className="py-2.5 px-3 text-right">SR</th>
                    <th className="py-2.5 px-3">Stage 1</th>
                    <th className="py-2.5 px-3">Stage 2 Class</th>
                    <th className="py-2.5 px-3 text-right">MOS</th>
                    <th className="py-2.5 px-3 text-right">P_OK</th>
                    <th className="py-2.5 px-3 text-right">Peak @</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70 text-xs">
                  {filteredRecords.map(({ sample, gate, peakTimestampSec }) => {
                    const isSelected = sample.id === activeEntry.sample.id;
                    const isPlaying = playingId === sample.id;
                    return (
                      <tr
                        key={sample.id}
                        onClick={() => setSelectedId(sample.id)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-sky-950/40'
                            : 'hover:bg-slate-800/50'
                        }`}
                      >
                        <td className="py-2.5 px-3">
                          <button
                            type="button"
                            onClick={(ev) => {
                              ev.stopPropagation();
                              handleTogglePlay(sample);
                            }}
                            aria-label={
                              isPlaying
                                ? `Stop ${sample.filename}`
                                : `Play ${sample.filename}`
                            }
                            className="p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors cursor-pointer"
                          >
                            {isPlaying ? (
                              <Square className="w-3.5 h-3.5 text-amber-400" />
                            ) : (
                              <Play className="w-3.5 h-3.5 text-sky-400" />
                            )}
                          </button>
                        </td>
                        <td className="py-2.5 px-3 font-mono text-slate-100">
                          <div className="truncate max-w-[190px]">{sample.filename}</div>
                          <div className="text-[11px] text-slate-400 font-sans">
                            Truth: {sample.groundTruth} · Legacy CNN: {sample.legacyPipelinePrediction}
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-300 whitespace-nowrap">
                          {(sample.sampleRateHz / 1000).toFixed(1)}k
                        </td>
                        <td className="py-2.5 px-3 font-mono font-semibold whitespace-nowrap">
                          {gate.stage1Status === 'OK' ? (
                            <span className="inline-flex items-center gap-1 text-emerald-400">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              OK
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-rose-400">
                              <AlertTriangle className="w-3.5 h-3.5" />
                              NG
                            </span>
                          )}
                        </td>
                        <td
                          className={`py-2.5 px-3 font-mono font-medium whitespace-nowrap ${
                            gate.finalClass === 'OK'
                              ? 'text-emerald-300'
                              : 'text-amber-300'
                          }`}
                        >
                          {gate.finalClass}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-200">
                          {sample.nisqa.mos_pred.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-200">
                          {sample.cnn.p_ok.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-300 whitespace-nowrap">
                          {gate.stage1Status === 'OK'
                            ? '—'
                            : `${peakTimestampSec.toFixed(2)}s`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="p-3.5 bg-slate-950/60 border-t border-slate-800 text-xs text-slate-400 flex items-center justify-between">
            <span>
              Resampling config:{' '}
              <code className="text-slate-200">
                {srMode === 48000
                  ? 'SR=48000 (FRAME_LEN=2400, HOP_LEN=1200, n_fft=2400)'
                  : "SR=44100 (res_type='soxr_vhq' sinc, FRAME_LEN=2205)"}
              </code>
            </span>
            <span className="font-mono tabular-nums text-slate-300">
              100% Offline Engine
            </span>
          </div>
        </div>

        {/* Right: Deep Stage 1 + Stage 2 Diagnostic Inspector for Selected File */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-lg p-5 flex flex-col justify-between space-y-5">
          <div>
            {/* File Header */}
            <div className="flex items-start justify-between gap-3 pb-4 border-b border-slate-800">
              <div className="min-w-0">
                <div className="text-xs font-mono text-slate-400">
                  {activeEntry.sample.id} · {activeEntry.sample.sampleRateHz} Hz Mono ·{' '}
                  {activeEntry.sample.durationSec}s
                </div>
                <h3 className="text-base font-semibold font-mono text-slate-100 truncate mt-0.5">
                  {activeEntry.sample.filename}
                </h3>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                  {activeEntry.sample.description}
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleTogglePlay(activeEntry.sample)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-sky-600 hover:bg-sky-500 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                {playingId === activeEntry.sample.id ? (
                  <>
                    <Square className="w-3.5 h-3.5" />
                    Stop Audio
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" />
                    Synthesize 48kHz
                  </>
                )}
              </button>
            </div>

            {/* Stage 1 & Stage 2 Decision Verdict Banner */}
            <div className="mt-4 p-3.5 rounded-lg bg-slate-950 border border-slate-800">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">
                  Stage 1 Gate Verdict:{' '}
                  <strong
                    className={
                      activeEntry.gate.stage1Status === 'OK'
                        ? 'text-emerald-400 font-mono'
                        : 'text-rose-400 font-mono'
                    }
                  >
                    {activeEntry.gate.stage1Status}
                  </strong>
                </span>
                <span className="text-slate-400">
                  Stage 2 Final Classification:{' '}
                  <strong className="text-sky-300 font-mono">
                    {activeEntry.gate.finalClass}
                  </strong>
                </span>
              </div>
              <div className="mt-2 text-xs text-slate-300 font-mono">
                {activeEntry.gate.vetoReasons.length === 0 ? (
                  <span className="text-emerald-400">
                    PASS: Passed all NISQA perceptual dimensions &amp; 1D CNN temporal checks.
                  </span>
                ) : (
                  <span className="text-amber-300">
                    VETO: {activeEntry.gate.vetoReasons.join(' | ')}
                  </span>
                )}
              </div>
            </div>

            {/* Path A: NISQA v2.0 5 Perceptual Dimensions */}
            <div className="mt-5">
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="font-semibold text-slate-200">
                  Path A: NISQA v2.0 Perceptual Dimensions (1.0 – 5.0 Scale)
                </span>
                <span className="font-mono text-slate-400 tabular-nums">
                  Pass: {activeEntry.gate.nisqaPass ? 'YES' : 'NO'}
                </span>
              </div>
              <div className="space-y-2">
                {[
                  {
                    label: 'mos_pred (Overall Quality)',
                    val: activeEntry.sample.nisqa.mos_pred,
                    min: thresholds.mosMin,
                  },
                  {
                    label: 'noi_pred (Noisiness / Hiss)',
                    val: activeEntry.sample.nisqa.noi_pred,
                    min: thresholds.noiMin,
                  },
                  {
                    label: 'dis_pred (Discontinuity / Cut)',
                    val: activeEntry.sample.nisqa.dis_pred,
                    min: thresholds.disMin,
                  },
                  {
                    label: 'col_pred (Coloration / Wobble)',
                    val: activeEntry.sample.nisqa.col_pred,
                    min: thresholds.colMin,
                  },
                  {
                    label: 'loud_pred (Loudness Stability)',
                    val: activeEntry.sample.nisqa.loud_pred,
                    min: 3.0,
                  },
                ].map((dim) => {
                  const pct = Math.min(100, Math.max(0, (dim.val / 5.0) * 100));
                  const pass = dim.val >= dim.min;
                  return (
                    <div key={dim.label} className="text-xs">
                      <div className="flex justify-between font-mono">
                        <span className="text-slate-300">{dim.label}</span>
                        <span
                          className={`tabular-nums ${
                            pass ? 'text-emerald-400' : 'text-rose-400 font-semibold'
                          }`}
                        >
                          {dim.val.toFixed(2)} / 5.00
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded overflow-hidden mt-1">
                        <div
                          className={`h-full ${
                            pass ? 'bg-emerald-500' : 'bg-rose-500'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Path B: 1D Dilated CNN 5-Way Softmax + Self-Similarity */}
            <div className="mt-5">
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="font-semibold text-slate-200">
                  Path B: 1D Dilated CNN Softmax + Acoustic Self-Similarity
                </span>
                <span className="font-mono text-slate-400 tabular-nums">
                  Self-Sim: {activeEntry.sample.selfSimilarityScore.toFixed(2)}
                </span>
              </div>
              <div className="grid grid-cols-5 gap-2 text-center font-mono text-xs">
                {[
                  { k: 'P_OK', v: activeEntry.sample.cnn.p_ok },
                  { k: 'P_Noise', v: activeEntry.sample.cnn.p_noise },
                  { k: 'P_Cut', v: activeEntry.sample.cnn.p_cut },
                  { k: 'P_Wobble', v: activeEntry.sample.cnn.p_wobble },
                  { k: 'P_Rep', v: activeEntry.sample.cnn.p_rep },
                ].map((item) => (
                  <div
                    key={item.k}
                    className="p-2 rounded bg-slate-950 border border-slate-800"
                  >
                    <div className="text-[11px] text-slate-400">{item.k}</div>
                    <div className="text-sm font-semibold text-slate-100 tabular-nums mt-0.5">
                      {(item.v * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 25ms AttentionLayer Activity Heatmap (`get_activity_heatmap`) */}
            <div className="mt-5">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="font-semibold text-slate-200">
                  AttentionLayer 25ms Activity Heatmap (event_timestamp)
                </span>
                <span className="font-mono text-amber-400 tabular-nums">
                  {activeEntry.gate.stage1Status === 'OK'
                    ? 'No Defect Peak'
                    : `Peak Defect @ ${activeEntry.peakTimestampSec.toFixed(3)}s`}
                </span>
              </div>
              <div className="h-14 bg-slate-950 border border-slate-800 rounded p-2 flex items-end gap-1">
                {activeEntry.sample.attentionHeatmap.map((weight, idx) => {
                  const isPeak =
                    activeEntry.gate.stage1Status === 'NG' &&
                    idx === activeEntry.peakFrameIndex;
                  const heightPct = Math.max(12, Math.round(weight * 100));
                  return (
                    <div
                      key={idx}
                      style={{ height: `${heightPct}%` }}
                      className={`flex-1 rounded-t transition-opacity ${
                        isPeak
                          ? 'bg-rose-500'
                          : weight > 0.55
                          ? 'bg-amber-400'
                          : 'bg-sky-500/60'
                      }`}
                      title={`Frame ${idx}: weight=${weight.toFixed(2)}`}
                    />
                  );
                })}
              </div>
            </div>
          </div>

          {/* 10-Element Fused X_gate Vector Footer */}
          <div className="pt-3 border-t border-slate-800">
            <div className="text-[11px] font-mono text-slate-400">
              10-Feature Stage 1 Vector X_gate:
            </div>
            <div className="mt-1 text-[11px] font-mono text-slate-300 bg-slate-950 px-2.5 py-1.5 rounded border border-slate-800 overflow-x-auto whitespace-nowrap tabular-nums">
              [{activeEntry.gate.xGateVector.map((n) => n.toFixed(2)).join(', ')}]
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
