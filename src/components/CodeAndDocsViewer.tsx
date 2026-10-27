import React, { useState, useEffect } from 'react';
import { Download, Copy, Check, FileCode, FileText } from 'lucide-react';

interface DeliverableFile {
  id: string;
  filename: string;
  path: string;
  category: 'Python Code' | 'Documentation' | 'Config';
  description: string;
}

export const DELIVERABLE_FILES: DeliverableFile[] = [
  {
    id: 'inference_py',
    filename: 'inference.py',
    path: '/downloads/inference.py',
    category: 'Python Code',
    description:
      'Production Two-Stage Cascaded Hybrid Gatekeeper (NISQA v2.0 + 1D Dilated CNN + Attention), soxr_vhq 48kHz/44.1kHz loader, 25ms peak timestamp extractor, and Excel (.xlsx) report generator.',
  },
  {
    id: 'train_hybrid_py',
    filename: 'train_hybrid.py',
    path: '/downloads/train_hybrid.py',
    category: 'Python Code',
    description:
      'Full 100% offline training pipeline for the 63-feature StandardScaler, 1D Temporal Dilated CNN (dilation rates 1,2,4,8) + AttentionLayer, and 10-feature Stage 1 XGBoost/Logistic Meta-Classifier.',
  },
  {
    id: 'gui_app_py',
    filename: 'gui_app.py',
    path: '/downloads/gui_app.py',
    category: 'Python Code',
    description:
      'Standalone 100% offline Python Desktop GUI (Tkinter / ttk) supporting single-file and folder batch inspection, live Stage 1 threshold controls, and Excel report export.',
  },
  {
    id: 'prerequisites_txt',
    filename: 'PREREQUISITES_INSTALLATION.txt',
    path: '/downloads/PREREQUISITES_INSTALLATION.txt',
    category: 'Documentation',
    description:
      'Step-by-step Prerequisites & Two-Phase 100% Offline Wheelhouse Installation Guide (`pip download` -> `pip install --no-index --find-links`) and directory structure.',
  },
  {
    id: 'user_manual_txt',
    filename: 'USER_MANUAL.txt',
    path: '/downloads/USER_MANUAL.txt',
    category: 'Documentation',
    description:
      'Detailed Operator & Engineer User Manual in .txt format explaining GUI usage, Stage 1/2 gate logic, Excel report columns, and threshold calibration.',
  },
  {
    id: 'readme_md',
    filename: 'README.md',
    path: '/downloads/README.md',
    category: 'Documentation',
    description:
      'Comprehensive project architecture documentation, tool evaluation benchmark table, Two-Stage Hybrid diagram, and CLI quickstart commands.',
  },
  {
    id: 'requirements_txt',
    filename: 'requirements.txt',
    path: '/downloads/requirements.txt',
    category: 'Config',
    description:
      'Pinned offline Python package versions (PyTorch, TensorFlow/Keras, Librosa, Soxr, SoundFile, XGBoost, OpenPyXL).',
  },
];

interface CodeAndDocsViewerProps {
  initialFileId?: string;
}

export const CodeAndDocsViewer: React.FC<CodeAndDocsViewerProps> = ({
  initialFileId = 'inference_py',
}) => {
  const [selectedId, setSelectedId] = useState<string>(initialFileId);
  const [fileContents, setFileContents] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<boolean>(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    if (initialFileId) {
      setSelectedId(initialFileId);
    }
  }, [initialFileId]);

  const activeFile =
    DELIVERABLE_FILES.find((f) => f.id === selectedId) || DELIVERABLE_FILES[0];

  useEffect(() => {
    let cancelled = false;
    async function loadFile() {
      if (fileContents[activeFile.id]) return;
      setLoading(true);
      try {
        const res = await fetch(activeFile.path);
        const text = await res.text();
        if (!cancelled) {
          setFileContents((prev) => ({ ...prev, [activeFile.id]: text }));
        }
      } catch {
        if (!cancelled) {
          setFileContents((prev) => ({
            ...prev,
            [activeFile.id]: '# Error loading file preview.',
          }));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadFile();
    return () => {
      cancelled = true;
    };
  }, [activeFile, fileContents]);

  const handleCopy = async () => {
    const content = fileContents[activeFile.id] || '';
    if (!content) return;
    await navigator.clipboard.writeText(content);
    setCopiedId(activeFile.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleDownloadFile = (file: DeliverableFile) => {
    const link = document.createElement('a');
    link.href = file.path;
    link.download = file.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadAll = () => {
    DELIVERABLE_FILES.forEach((f, idx) => {
      setTimeout(() => handleDownloadFile(f), idx * 180);
    });
  };

  const currentText = fileContents[activeFile.id] || '';
  const lineCount = currentText ? currentText.split('\n').length : 0;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-slate-800">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">
            100% Offline Production Code & Documentation Suite
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Inspect, copy, or download the complete Python training, inference, desktop Tkinter GUI, prerequisites guide (.txt), user manual (.txt), and README.md files.
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={handleDownloadAll}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-sky-600 hover:bg-sky-500 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Download All 7 Offline Files
          </button>
        </div>
      </div>

      {/* File Selector Tabs + Viewer Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: File List */}
        <div className="lg:col-span-4 space-y-2">
          {DELIVERABLE_FILES.map((file) => {
            const isSelected = file.id === activeFile.id;
            return (
              <div
                key={file.id}
                className={`p-3.5 rounded-lg border transition-colors ${
                  isSelected
                    ? 'bg-slate-800/90 border-sky-500/60 text-slate-100'
                    : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => setSelectedId(file.id)}
                    className="flex items-center gap-2.5 text-left font-mono text-sm font-medium text-slate-100 hover:text-sky-400 transition-colors truncate cursor-pointer"
                  >
                    {file.category === 'Python Code' ? (
                      <FileCode className="w-4 h-4 text-sky-400 shrink-0" />
                    ) : (
                      <FileText className="w-4 h-4 text-emerald-400 shrink-0" />
                    )}
                    <span className="truncate">{file.filename}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDownloadFile(file)}
                    aria-label={`Download ${file.filename}`}
                    className="px-2.5 py-1 text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    Download
                  </button>
                </div>
                <p
                  onClick={() => setSelectedId(file.id)}
                  className="text-xs text-slate-400 mt-2 leading-relaxed cursor-pointer"
                >
                  {file.description}
                </p>
                <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-500">
                  <span>{file.category}</span>
                  <span aria-hidden="true">·</span>
                  <span>100% Offline Ready</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Right Column: Full Source Code / Document Viewer */}
        <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 bg-slate-950/70 border-b border-slate-800">
            <div className="flex items-center gap-3 min-w-0">
              <span className="font-mono text-sm font-semibold text-slate-100 truncate">
                {activeFile.filename}
              </span>
              <span className="text-xs text-slate-500 font-mono tabular-nums whitespace-nowrap">
                {lineCount} lines
              </span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                {copiedId === activeFile.id ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    Copy File Content
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={() => handleDownloadFile(activeFile)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-sky-600 hover:bg-sky-500 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                Download {activeFile.filename}
              </button>
            </div>
          </div>

          <div className="p-4 max-h-[640px] overflow-y-auto bg-[#070B14]">
            {loading ? (
              <div className="py-12 text-center text-sm text-slate-400">
                Loading {activeFile.filename}...
              </div>
            ) : (
              <pre className="text-xs font-mono text-slate-200 leading-relaxed overflow-x-auto whitespace-pre">
                {currentText}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
