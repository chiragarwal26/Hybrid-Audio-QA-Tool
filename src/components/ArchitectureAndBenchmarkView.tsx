import React from 'react';

interface CandidateTool {
  name: string;
  intrusiveness: string;
  maxSampleRate: string;
  pythonBatch: string;
  license: string;
  verdict: string;
  status: 'winner' | 'disqualified' | 'suboptimal';
}

const CANDIDATE_TOOLS: CandidateTool[] = [
  {
    name: 'NISQA (v2.0) + 1D Dilated CNN',
    intrusiveness: 'Non-Intrusive (No reference)',
    maxSampleRate: '48 kHz (Fullband)',
    pythonBatch: 'Yes (Native PyTorch + Keras)',
    license: 'Open-Source (MIT)',
    verdict:
      'WINNER: Built for 48 kHz; outputs Overall MOS + 4 diagnostic sub-scores paired with 1D CNN temporal repetition & 25ms attention peak.',
    status: 'winner',
  },
  {
    name: 'PESQ (ITU-T P.862)',
    intrusiveness: 'Intrusive (Needs clean reference)',
    maxSampleRate: '16 kHz (Wideband)',
    pythonBatch: 'Yes (pesq library)',
    license: 'Proprietary ITU license',
    verdict: 'Disqualified: Requires a golden reference file and is blind to frequencies above 8 kHz.',
    status: 'disqualified',
  },
  {
    name: 'ViSQOL (v3)',
    intrusiveness: 'Intrusive (Needs clean reference)',
    maxSampleRate: '48 kHz (Audio) / 16 kHz (Speech)',
    pythonBatch: 'Yes (Python wrapper / C++)',
    license: 'Open-Source (Apache 2.0)',
    verdict: 'Disqualified: Strictly full-reference; cannot evaluate standalone test recordings.',
    status: 'disqualified',
  },
  {
    name: 'ITU-T P.563',
    intrusiveness: 'Non-Intrusive (No reference)',
    maxSampleRate: '8 kHz (Narrowband)',
    pythonBatch: 'External ANSI-C binary wrapper',
    license: 'Proprietary ITU license',
    verdict: 'Disqualified: Hardcoded for 8 kHz telephony (300–3400 Hz); erases 83% of a 48 kHz spectrum.',
    status: 'disqualified',
  },
  {
    name: 'GL Comm. VQT (P.563)',
    intrusiveness: 'Non-Intrusive (via P.563 plugin)',
    maxSampleRate: '8 kHz (for P.563)',
    pythonBatch: 'No (Proprietary GUI / CLI SDK)',
    license: 'Commercial Hardware/Dongle',
    verdict: 'Disqualified: Uses the same 8 kHz P.563 engine under the hood; heavyweight commercial lock-in.',
    status: 'disqualified',
  },
  {
    name: 'Sevana AQuA',
    intrusiveness: 'Intrusive (Needs clean reference)',
    maxSampleRate: '48 kHz',
    pythonBatch: 'CLI / C++ / Python SDK',
    license: 'Commercial Per-Seat/Server',
    verdict: 'Disqualified: Requires a reference waveform for comparison.',
    status: 'disqualified',
  },
  {
    name: 'Sevana PVQA (PQUA)',
    intrusiveness: 'Non-Intrusive (No reference)',
    maxSampleRate: '8 kHz / 16 kHz (Standard)',
    pythonBatch: 'C++ / Python SDK',
    license: 'Commercial License',
    verdict: 'Sub-optimal: Tuned for VoIP/VoLTE telecom impairments; closed-source licensing.',
    status: 'suboptimal',
  },
  {
    name: 'SpeechMetrics',
    intrusiveness: 'Mixed (Mostly Intrusive)',
    maxSampleRate: '16 kHz (for MOSNet / SRMR)',
    pythonBatch: 'Yes (Python wrapper)',
    license: 'Open-Source',
    verdict: 'Sub-optimal: Only non-intrusive metrics are MOSNet (16 kHz) and SRMR (measures reverb, not cuts/wobbles).',
    status: 'suboptimal',
  },
];

export const ArchitectureAndBenchmarkView: React.FC = () => {
  return (
    <div className="space-y-8">
      {/* Section 1: Two-Stage Cascaded Architecture Diagram */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-slate-800">
          <div>
            <h2 className="text-lg font-semibold text-slate-100">
              01. Two-Stage Cascaded Hybrid Production Architecture
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Decouples binary OK vs. NG segregation (Stage 1) from 4-class NG defect resolution and 25ms peak timestamping (Stage 2).
            </p>
          </div>
          <a
            href="/assets/aistudio/Hybrid_Audio_QA_Pipeline_Design_b3c6df16.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="px-3.5 py-2 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            View Attached Architecture PDF
          </a>
        </div>

        {/* Visual Flow Grid */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Path A & Path B Dual Feature Extractors */}
          <div className="lg:col-span-5 flex flex-col justify-between gap-4">
            <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800">
              <div className="text-xs font-mono text-sky-400">PATH A · PERCEPTUAL BRANCH</div>
              <h3 className="text-sm font-semibold text-slate-100 mt-1">
                NISQA v2.0 (Native 48 kHz Fullband)
              </h3>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                Processes untouched 48 kHz audio using Kaiser-windowed polyphase filtering. Outputs 5 human perceptual dimensions on a 1.0–5.0 scale:
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs font-mono text-slate-300">
                <div>1. mos_pred (Overall)</div>
                <div>2. noi_pred (Noisiness)</div>
                <div>3. dis_pred (Discontinuity)</div>
                <div>4. col_pred (Coloration)</div>
                <div className="col-span-2">5. loud_pred (Loudness stability)</div>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800">
              <div className="text-xs font-mono text-amber-400">PATH B · TEMPORAL & PHYSICS BRANCH</div>
              <h3 className="text-sm font-semibold text-slate-100 mt-1">
                63-Feature Physics Stack + 1D Dilated CNN + Attention
              </h3>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                Uses <code className="text-slate-200">res_type=&apos;soxr_vhq&apos;</code> sinc interpolation (zero artificial wobble) or native 48 kHz frames (<code className="text-slate-200">FRAME_LEN=2400</code>). Dilation rates (1, 2, 4, 8) + Acoustic Self-Similarity catch clean phrase loops (<code className="text-slate-200">NG_Rep</code>).
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs font-mono text-slate-300">
                <div>• 5 Softmax Probabilities</div>
                <div>• 25ms Attention Heatmap</div>
                <div className="col-span-2">• Acoustic Self-Similarity Matrix Score</div>
              </div>
            </div>
          </div>

          {/* Stage 1 & Stage 2 Gatekeeper Flow */}
          <div className="lg:col-span-7 flex flex-col justify-between gap-4">
            <div className="p-5 rounded-lg bg-slate-950/90 border border-sky-500/40">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-mono text-sky-400">
                  STAGE 1 · ROBUST BINARY GATEKEEPER (OK vs. NG SEGREGATION)
                </span>
                <span className="text-xs font-mono text-emerald-400 tabular-nums">
                  Target Binary Accuracy: 97–99%
                </span>
              </div>
              <h3 className="text-base font-semibold text-slate-100 mt-1">
                10-Feature Consensus Vector Fusion
              </h3>
              <p className="text-xs font-mono text-slate-300 mt-2 bg-slate-900 p-2.5 rounded border border-slate-800 overflow-x-auto">
                X_gate = [MOS, NOI, DIS, COL, LOUD, P_OK, P_Cut, P_Noise, P_Rep, P_Wobble]
              </p>
              <div className="mt-3 space-y-1.5 text-xs text-slate-300">
                <p>
                  <strong className="text-slate-100">Perceptual Gate (NISQA):</strong>{' '}
                  <code className="text-sky-300">mos_pred &gt;= 3.6 AND noi_pred &gt;= 3.5 AND dis_pred &gt;= 3.5 AND col_pred &gt;= 3.5</code>
                </p>
                <p>
                  <strong className="text-slate-100">Temporal Gate (1D CNN):</strong>{' '}
                  <code className="text-sky-300">P_OK &gt;= 0.35 AND max(P_Rep, P_Cut, P_Noise) &lt; 0.55</code>
                </p>
              </div>
            </div>

            <div className="p-5 rounded-lg bg-slate-950/90 border border-slate-800">
              <div className="text-xs font-mono text-rose-400">
                STAGE 2 · NG DEFECT SUB-CLASSIFIER & 25MS TIMESTAMP RESOLVER
              </div>
              <h3 className="text-base font-semibold text-slate-100 mt-1">
                Zeroes Out P_OK & Resolves Exact Defect Dimension
              </h3>
              <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div className="font-mono font-semibold text-slate-100">NG_Noise</div>
                  <div className="text-slate-400 mt-0.5">High CNN P_Noise + Low NISQA noi_pred</div>
                </div>
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div className="font-mono font-semibold text-slate-100">NG_Cut</div>
                  <div className="text-slate-400 mt-0.5">High CNN P_Cut + Low NISQA dis_pred</div>
                </div>
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div className="font-mono font-semibold text-slate-100">NG_Wobble</div>
                  <div className="text-slate-400 mt-0.5">High CNN P_Wobble + Low NISQA col_pred</div>
                </div>
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div className="font-mono font-semibold text-slate-100">NG_Rep</div>
                  <div className="text-slate-400 mt-0.5">High CNN P_Rep + Self-Similarity Peak</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Section 2: Candidate Tool & Standard Evaluation Matrix */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
        <div className="p-6 border-b border-slate-800">
          <h2 className="text-lg font-semibold text-slate-100">
            02. Thorough Evaluation of Candidate Tools & Standards (48 kHz Non-Intrusive Requirement)
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Why 6 of the 7 standalone standards fail 48 kHz single-ended production constraints, and why NISQA v2.0 + 1D Dilated CNN wins.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-950/60 text-xs font-medium text-slate-400">
                <th className="py-3 px-4">Tool / Standard</th>
                <th className="py-3 px-4">Intrusiveness (Reference Needed?)</th>
                <th className="py-3 px-4">Max Sample Rate</th>
                <th className="py-3 px-4">Native Python Batch</th>
                <th className="py-3 px-4">License / Cost</th>
                <th className="py-3 px-4">Verdict for 48 kHz Pipeline</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 text-xs">
              {CANDIDATE_TOOLS.map((tool) => (
                <tr
                  key={tool.name}
                  className={
                    tool.status === 'winner'
                      ? 'bg-emerald-950/20 hover:bg-emerald-950/30 transition-colors'
                      : 'hover:bg-slate-800/40 transition-colors'
                  }
                >
                  <td className="py-3.5 px-4 font-semibold text-slate-100 whitespace-nowrap">
                    {tool.name}
                  </td>
                  <td className="py-3.5 px-4 text-slate-300">{tool.intrusiveness}</td>
                  <td className="py-3.5 px-4 font-mono tabular-nums text-slate-200 whitespace-nowrap">
                    {tool.maxSampleRate}
                  </td>
                  <td className="py-3.5 px-4 text-slate-300">{tool.pythonBatch}</td>
                  <td className="py-3.5 px-4 text-slate-400">{tool.license}</td>
                  <td
                    className={`py-3.5 px-4 leading-relaxed ${
                      tool.status === 'winner'
                        ? 'text-emerald-300 font-medium'
                        : tool.status === 'disqualified'
                        ? 'text-rose-300'
                        : 'text-amber-300'
                    }`}
                  >
                    {tool.verdict}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
