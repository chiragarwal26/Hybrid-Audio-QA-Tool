/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Download } from 'lucide-react';
import { InteractiveWorkbench } from './components/InteractiveWorkbench';
import { CodeAndDocsViewer, DELIVERABLE_FILES } from './components/CodeAndDocsViewer';
import { ArchitectureAndBenchmarkView } from './components/ArchitectureAndBenchmarkView';

type ActiveSection = 'workbench' | 'python_code' | 'docs_manual' | 'architecture';

export default function App() {
  const [activeSection, setActiveSection] = useState<ActiveSection>('workbench');
  const [initialFileId, setInitialFileId] = useState<string>('inference_py');

  const handleOpenCodeTab = (fileId: string) => {
    setInitialFileId(fileId);
    setActiveSection('python_code');
  };

  const handleDownloadAllSuite = () => {
    DELIVERABLE_FILES.forEach((f, idx) => {
      setTimeout(() => {
        const link = document.createElement('a');
        link.href = f.path;
        link.download = f.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }, idx * 180);
    });
  };

  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-100 flex flex-col">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="flex items-center justify-between gap-8 px-6 py-4 border-b border-slate-800 bg-slate-950/80">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveSection('workbench');
          }}
          className="text-lg font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0"
        >
          HybridAudioQA
        </a>

        {/* Zone 2: 4 concise single-line text navigation links */}
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-300">
          <a
            href="#workbench"
            onClick={(e) => {
              e.preventDefault();
              setActiveSection('workbench');
            }}
            className={`hover:text-white hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0 ${
              activeSection === 'workbench' ? 'text-white underline' : ''
            }`}
          >
            Interactive Workbench
          </a>
          <a
            href="#python-code"
            onClick={(e) => {
              e.preventDefault();
              handleOpenCodeTab('inference_py');
            }}
            className={`hover:text-white hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0 ${
              activeSection === 'python_code' ? 'text-white underline' : ''
            }`}
          >
            Python Code
          </a>
          <a
            href="#docs-manual"
            onClick={(e) => {
              e.preventDefault();
              setInitialFileId('prerequisites_txt');
              setActiveSection('docs_manual');
            }}
            className={`hover:text-white hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0 ${
              activeSection === 'docs_manual' ? 'text-white underline' : ''
            }`}
          >
            Offline Guides
          </a>
          <a
            href="#architecture"
            onClick={(e) => {
              e.preventDefault();
              setActiveSection('architecture');
            }}
            className={`hover:text-white hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0 ${
              activeSection === 'architecture' ? 'text-white underline' : ''
            }`}
          >
            Architecture
          </a>
        </nav>

        {/* Zone 3: 1 primary action */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={handleDownloadAllSuite}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-sky-600 rounded-lg hover:bg-sky-500 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            Download Offline Suite
          </button>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 max-w-[1380px] w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Hero / Quick Deliverable Access Banner */}
        <section className="bg-slate-900 border border-slate-800 rounded-lg p-5">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5">
            <div className="max-w-3xl">
              <div className="text-xs text-slate-400">
                48 kHz Fullband Non-Intrusive Audio QA · NISQA v2.0 + 1D Temporal Dilated CNN + Attention · 100% Offline
              </div>
              <h1 className="text-xl sm:text-2xl font-bold text-slate-100 mt-1 text-balance">
                Two-Stage Cascaded Hybrid Audio QA Workbench &amp; Complete Offline Python Deliverable Package
              </h1>
              <p className="text-xs sm:text-sm text-slate-300 mt-1.5 leading-relaxed">
                Segregates up to 48 kHz mono recordings into <strong className="text-emerald-400">OK</strong> vs. <strong className="text-rose-400">NG</strong> via a 10-feature Consensus Gate, resolves NG defects into <code className="text-slate-200">NG_Noise</code>, <code className="text-slate-200">NG_Cut</code>, <code className="text-slate-200">NG_Wobble</code>, and <code className="text-slate-200">NG_Rep</code>, and pinpoints 25ms peak defect timestamps.
              </p>
            </div>

            {/* Direct Quick-Open Buttons for Requested Deliverables */}
            <div className="flex flex-wrap items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={() => handleOpenCodeTab('train_hybrid_py')}
                className="px-3 py-2 text-xs font-mono font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                train_hybrid.py
              </button>
              <button
                type="button"
                onClick={() => handleOpenCodeTab('inference_py')}
                className="px-3 py-2 text-xs font-mono font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                inference.py
              </button>
              <button
                type="button"
                onClick={() => handleOpenCodeTab('gui_app_py')}
                className="px-3 py-2 text-xs font-mono font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                gui_app.py
              </button>
              <button
                type="button"
                onClick={() => {
                  setInitialFileId('prerequisites_txt');
                  setActiveSection('docs_manual');
                }}
                className="px-3 py-2 text-xs font-mono font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                PREREQUISITES_INSTALLATION.txt
              </button>
              <button
                type="button"
                onClick={() => {
                  setInitialFileId('user_manual_txt');
                  setActiveSection('docs_manual');
                }}
                className="px-3 py-2 text-xs font-mono font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                USER_MANUAL.txt
              </button>
              <button
                type="button"
                onClick={() => {
                  setInitialFileId('readme_md');
                  setActiveSection('docs_manual');
                }}
                className="px-3 py-2 text-xs font-mono font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                README.md
              </button>
            </div>
          </div>

          {/* Interactive Workspace Mode Switcher */}
          <div className="mt-4 pt-4 border-t border-slate-800 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setActiveSection('workbench')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeSection === 'workbench'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              01. Live Hybrid Gatekeeper GUI Workbench
            </button>
            <button
              type="button"
              onClick={() => handleOpenCodeTab('inference_py')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeSection === 'python_code'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              02. Full Python Training, Inference &amp; GUI Code
            </button>
            <button
              type="button"
              onClick={() => {
                setInitialFileId('prerequisites_txt');
                setActiveSection('docs_manual');
              }}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeSection === 'docs_manual'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              03. Offline Installation (.txt), User Manual (.txt) &amp; README.md
            </button>
            <button
              type="button"
              onClick={() => setActiveSection('architecture')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeSection === 'architecture'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              04. Architecture &amp; Tool Evaluation Benchmark
            </button>
          </div>
        </section>

        {/* Active View Content */}
        {activeSection === 'workbench' && <InteractiveWorkbench />}
        {(activeSection === 'python_code' || activeSection === 'docs_manual') && (
          <CodeAndDocsViewer initialFileId={initialFileId} />
        )}
        {activeSection === 'architecture' && <ArchitectureAndBenchmarkView />}
      </main>

      {/* Clean Quiet Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-6 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <span>
          NISQA v2.0 + 1D Temporal Dilated CNN Hybrid Audio QA Pipeline · 100% Offline Non-Intrusive Fullband Solution
        </span>
        <div className="flex items-center gap-4">
          <a
            href="/downloads/PREREQUISITES_INSTALLATION.txt"
            download="PREREQUISITES_INSTALLATION.txt"
            className="hover:text-slate-300 transition-colors"
          >
            PREREQUISITES_INSTALLATION.txt
          </a>
          <a
            href="/downloads/USER_MANUAL.txt"
            download="USER_MANUAL.txt"
            className="hover:text-slate-300 transition-colors"
          >
            USER_MANUAL.txt
          </a>
          <a
            href="/downloads/README.md"
            download="README.md"
            className="hover:text-slate-300 transition-colors"
          >
            README.md
          </a>
        </div>
      </footer>
    </div>
  );
}
