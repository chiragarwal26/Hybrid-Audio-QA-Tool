#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HYBRID AUDIO QA PIPELINE - 100% OFFLINE DESKTOP GUI WORKBENCH (gui_app.py)
Architecture: NISQA v2.0 (48 kHz Fullband) + 1D Temporal Dilated CNN + Attention
Framework: Native Python Tkinter / ttk (Zero Browser or Cloud Dependency)
================================================================================
Features:
  - Single-File & Multi-Threaded Batch Folder Audio QA Inspection (.wav / .flac / .mp3)
  - Live Stage 1 Threshold Sliders (MOS, NOI, DIS, COL, P_OK, Max Defect Cap)
  - Switch between "Rule-Based Consensus Gate (Zero Retraining)" and "XGBoost Meta-Gate"
  - Switch between "44.1 kHz Baseline (soxr_vhq Zero-Retrain Fix)" and "48 kHz Native"
  - Real-time Treeview Table showing Stage 1 Verdict (OK/NG), Stage 2 Defect Class,
    25ms Peak Event Timestamp (s), all 5 NISQA scores, and all 5 CNN probabilities
  - One-click Export to Color-Coded Excel Report (`Audio_QA_Hybrid_Report.xlsx`)
"""

import os
import sys
import glob
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from inference import HybridAudioQAPipeline, export_results_to_excel, DEFAULT_GATE_CONFIG


class OfflineAudioQAGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NISQA v2.0 + 1D Dilated CNN Hybrid Audio QA Workbench (100% Offline - Up to 48 kHz)")
        self.geometry("1360x820")
        self.minsize(1180, 720)
        self.configure(bg="#0F172A")

        self.results_data = []
        self.worker_queue = queue.Queue()
        self.is_running = False

        # Control Variables
        self.weights_dir_var = tk.StringVar(value="./weights")
        self.input_path_var = tk.StringVar(value="")
        self.sr_mode_var = tk.IntVar(value=44100)
        self.gate_mode_var = tk.StringVar(value="rule_consensus")

        self.mos_min_var = tk.DoubleVar(value=DEFAULT_GATE_CONFIG["mos_min"])
        self.noi_min_var = tk.DoubleVar(value=DEFAULT_GATE_CONFIG["noi_min"])
        self.dis_min_var = tk.DoubleVar(value=DEFAULT_GATE_CONFIG["dis_min"])
        self.col_min_var = tk.DoubleVar(value=DEFAULT_GATE_CONFIG["col_min"])
        self.p_ok_min_var = tk.DoubleVar(value=DEFAULT_GATE_CONFIG["p_ok_min"])
        self.defect_cap_var = tk.DoubleVar(value=DEFAULT_GATE_CONFIG["max_defect_cap"])

        self.status_var = tk.StringVar(value="Ready (100% Offline Air-Gapped Mode) | Select a file or folder to inspect.")
        self.summary_var = tk.StringVar(value="Total: 0  |  OK: 0  |  NG: 0  |  NG_Noise: 0  ·  NG_Cut: 0  ·  NG_Wobble: 0  ·  NG_Rep: 0")

        self._configure_styles()
        self._build_layout()
        self.after(100, self._poll_worker_queue)

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0F172A")
        style.configure("Card.TFrame", background="#1E293B", relief="flat")
        style.configure("TLabel", background="#0F172A", foreground="#F8FAFC", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#1E293B", foreground="#E2E8F0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#0F172A", foreground="#F8FAFC", font=("Segoe UI", 15, "bold"))
        style.configure("Primary.TButton", background="#0284C7", foreground="#FFFFFF", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Primary.TButton", background=[("active", "#0369A1")])
        style.configure("Secondary.TButton", background="#334155", foreground="#F8FAFC", font=("Segoe UI", 10), padding=6)
        style.map("Secondary.TButton", background=[("active", "#475569")])

        style.configure(
            "Treeview",
            background="#0F172A",
            foreground="#F1F5F9",
            fieldbackground="#0F172A",
            rowheight=28,
            font=("Consolas", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#1E293B",
            foreground="#94A3B8",
            font=("Segoe UI", 9, "bold"),
        )

    def _build_layout(self):
        # Top Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", padx=16, pady=(14, 8))
        ttk.Label(
            header_frame,
            text="Hybrid Audio QA Pipeline — NISQA v2.0 (48 kHz) + 1D Temporal Dilated CNN + Attention",
            style="Header.TLabel",
        ).pack(side="left")
        ttk.Label(
            header_frame,
            text="Non-Intrusive (Zero Reference) · 100% Offline",
            foreground="#38BDF8",
        ).pack(side="right")

        # Configuration & Threshold Panel
        cfg_frame = ttk.Frame(self, style="Card.TFrame", padding=12)
        cfg_frame.pack(fill="x", padx=16, pady=6)

        row1 = ttk.Frame(cfg_frame, style="Card.TFrame")
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Input Audio / Folder:", style="Card.TLabel").pack(side="left", padx=(0, 6))
        ttk.Entry(row1, textvariable=self.input_path_var, width=52).pack(side="left", padx=4)
        ttk.Button(row1, text="Select File", style="Secondary.TButton", command=self._browse_file).pack(side="left", padx=4)
        ttk.Button(row1, text="Select Folder", style="Secondary.TButton", command=self._browse_folder).pack(side="left", padx=4)

        ttk.Label(row1, text="   CNN Sample Rate:", style="Card.TLabel").pack(side="left", padx=(12, 4))
        ttk.Radiobutton(row1, text="44.1 kHz (soxr_vhq fix)", variable=self.sr_mode_var, value=44100).pack(side="left", padx=4)
        ttk.Radiobutton(row1, text="48 kHz Native", variable=self.sr_mode_var, value=48000).pack(side="left", padx=4)

        ttk.Button(row1, text="Export Excel (.xlsx)", style="Secondary.TButton", command=self._export_excel).pack(side="right", padx=4)
        ttk.Button(row1, text="Run Hybrid QA Pipeline", style="Primary.TButton", command=self._start_inspection).pack(side="right", padx=6)

        # Row 2: Stage 1 Consensus Thresholds
        row2 = ttk.Frame(cfg_frame, style="Card.TFrame")
        row2.pack(fill="x", pady=(8, 2))
        ttk.Label(row2, text="Stage 1 GateThresholds:", style="Card.TLabel").pack(side="left", padx=(0, 8))

        for label_txt, var in [
            ("MOS >=", self.mos_min_var),
            ("NOI >=", self.noi_min_var),
            ("DIS >=", self.dis_min_var),
            ("COL >=", self.col_min_var),
            ("CNN P_OK >=", self.p_ok_min_var),
            ("Max Defect <", self.defect_cap_var),
        ]:
            ttk.Label(row2, text=label_txt, style="Card.TLabel").pack(side="left", padx=(6, 2))
            ttk.Spinbox(row2, textvariable=var, from_=0.05, to=4.95, increment=0.05, width=6).pack(side="left", padx=(0, 6))

        ttk.Label(row2, text="Gate Mode:", style="Card.TLabel").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            row2,
            textvariable=self.gate_mode_var,
            values=["rule_consensus", "xgb_meta"],
            state="readonly",
            width=15,
        ).pack(side="left")

        # Summary Strip
        sum_frame = ttk.Frame(self)
        sum_frame.pack(fill="x", padx=16, pady=(6, 4))
        ttk.Label(sum_frame, textvariable=self.summary_var, font=("Consolas", 10, "bold"), foreground="#E2E8F0").pack(side="left")

        # Results Treeview Table
        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=16, pady=6)

        cols = (
            "Filename",
            "Native_SR_Hz",
            "Stage1_Binary_Verdict",
            "Stage2_Final_Class",
            "Event_Timestamp_Sec",
            "NISQA_MOS",
            "NISQA_NOI",
            "NISQA_DIS",
            "NISQA_COL",
            "CNN_P_OK",
            "CNN_P_Noise",
            "CNN_P_Cut",
            "CNN_P_Wobble",
            "CNN_P_Rep",
            "Gate_Veto_Reason",
        )
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        headings = {
            "Filename": ("Audio File", 210),
            "Native_SR_Hz": ("SR (Hz)", 70),
            "Stage1_Binary_Verdict": ("Stage 1", 70),
            "Stage2_Final_Class": ("Final Class", 100),
            "Event_Timestamp_Sec": ("Peak @ (s)", 80),
            "NISQA_MOS": ("MOS", 60),
            "NISQA_NOI": ("NOI", 60),
            "NISQA_DIS": ("DIS", 60),
            "NISQA_COL": ("COL", 60),
            "CNN_P_OK": ("P_OK", 65),
            "CNN_P_Noise": ("P_Noise", 70),
            "CNN_P_Cut": ("P_Cut", 65),
            "CNN_P_Wobble": ("P_Wobble", 75),
            "CNN_P_Rep": ("P_Rep", 65),
            "Gate_Veto_Reason": ("Stage 1 Consensus / Veto Diagnostics", 320),
        }
        for col_id, (title, width) in headings.items():
            self.tree.heading(col_id, text=title)
            self.tree.column(col_id, width=width, anchor="w" if col_id in ("Filename", "Gate_Veto_Reason") else "center")

        self.tree.tag_configure("OK_ROW", foreground="#4ADE80")
        self.tree.tag_configure("NG_ROW", foreground="#F87171")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Bottom Status Bar
        status_bar = ttk.Frame(self, style="Card.TFrame", padding=6)
        status_bar.pack(fill="x", padx=16, pady=(2, 12))
        ttk.Label(status_bar, textvariable=self.status_var, style="Card.TLabel").pack(side="left")

    def _browse_file(self):
        fpath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.flac *.mp3"), ("All Files", "*.*")],
        )
        if fpath:
            self.input_path_var.set(fpath)

    def _browse_folder(self):
        dpath = filedialog.askdirectory(title="Select Folder Containing Audio Files")
        if dpath:
            self.input_path_var.set(dpath)

    def _start_inspection(self):
        if self.is_running:
            return
        target = self.input_path_var.get().strip()
        if not target or not os.path.exists(target):
            messagebox.showwarning("Input Required", "Please select a valid audio file or folder first.")
            return

        files = []
        if os.path.isfile(target):
            files = [target]
        else:
            for ext in ("*.wav", "*.flac", "*.mp3", "*.WAV", "*.FLAC"):
                files.extend(glob.glob(os.path.join(target, "**", ext), recursive=True))
            files = sorted(set(files))

        if not files:
            messagebox.showwarning("No Audio Files", "No .wav, .flac, or .mp3 files found in the selected path.")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.results_data.clear()
        self.is_running = True
        self.status_var.set(f"Inspecting {len(files)} audio file(s)...")

        gate_cfg = {
            "mos_min": float(self.mos_min_var.get()),
            "noi_min": float(self.noi_min_var.get()),
            "dis_min": float(self.dis_min_var.get()),
            "col_min": float(self.col_min_var.get()),
            "p_ok_min": float(self.p_ok_min_var.get()),
            "max_defect_cap": float(self.defect_cap_var.get()),
            "gate_mode": self.gate_mode_var.get(),
        }
        sr_val = int(self.sr_mode_var.get())
        weights_dir = self.weights_dir_var.get().strip()

        threading.Thread(
            target=self._worker_run,
            args=(files, weights_dir, sr_val, gate_cfg),
            daemon=True,
        ).start()

    def _worker_run(self, files, weights_dir, sr_val, gate_cfg):
        try:
            engine = HybridAudioQAPipeline(weights_dir=weights_dir, cnn_sr=sr_val, gate_config=gate_cfg)
            for idx, fpath in enumerate(files, 1):
                row = engine.inspect_file(fpath)
                self.worker_queue.put(("ROW", idx, len(files), row))
            self.worker_queue.put(("DONE", len(files)))
        except Exception as exc:
            self.worker_queue.put(("ERROR", str(exc)))

    def _poll_worker_queue(self):
        try:
            while True:
                msg = self.worker_queue.get_nowait()
                kind = msg[0]
                if kind == "ROW":
                    _, idx, total, row = msg
                    self.results_data.append(row)
                    tag = "OK_ROW" if row["Stage1_Binary_Verdict"] == "OK" else "NG_ROW"
                    self.tree.insert(
                        "",
                        "end",
                        values=(
                            row["Filename"],
                            row["Native_SR_Hz"],
                            row["Stage1_Binary_Verdict"],
                            row["Stage2_Final_Class"],
                            f"{row['Event_Timestamp_Sec']:.3f}",
                            f"{row['NISQA_MOS']:.2f}",
                            f"{row['NISQA_NOI']:.2f}",
                            f"{row['NISQA_DIS']:.2f}",
                            f"{row['NISQA_COL']:.2f}",
                            f"{row['CNN_P_OK']:.2f}",
                            f"{row['CNN_P_Noise']:.2f}",
                            f"{row['CNN_P_Cut']:.2f}",
                            f"{row['CNN_P_Wobble']:.2f}",
                            f"{row['CNN_P_Rep']:.2f}",
                            row["Gate_Veto_Reason"],
                        ),
                        tags=(tag,),
                    )
                    self.status_var.set(f"Processed [{idx}/{total}]: {row['Filename']} -> {row['Stage2_Final_Class']}")
                    self._update_summary()
                elif kind == "DONE":
                    self.is_running = False
                    self.status_var.set(f"Completed batch inspection of {msg[1]} audio files (100% Offline).")
                elif kind == "ERROR":
                    self.is_running = False
                    self.status_var.set(f"Error: {msg[1]}")
                    messagebox.showerror("Pipeline Error", msg[1])
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_worker_queue)

    def _update_summary(self):
        tot = len(self.results_data)
        ok_c = sum(1 for r in self.results_data if r["Stage1_Binary_Verdict"] == "OK")
        ng_c = tot - ok_c
        c_noi = sum(1 for r in self.results_data if r["Stage2_Final_Class"] == "NG_Noise")
        c_cut = sum(1 for r in self.results_data if r["Stage2_Final_Class"] == "NG_Cut")
        c_wob = sum(1 for r in self.results_data if r["Stage2_Final_Class"] == "NG_Wobble")
        c_rep = sum(1 for r in self.results_data if r["Stage2_Final_Class"] == "NG_Rep")
        self.summary_var.set(
            f"Total: {tot}  |  OK: {ok_c}  |  NG: {ng_c}  |  "
            f"NG_Noise: {c_noi}  ·  NG_Cut: {c_cut}  ·  NG_Wobble: {c_wob}  ·  NG_Rep: {c_rep}"
        )

    def _export_excel(self):
        if not self.results_data:
            messagebox.showinfo("No Results", "Run the Hybrid QA Pipeline first before exporting to Excel.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save Hybrid Audio QA Excel Report",
            defaultextension=".xlsx",
            initialfile="Audio_QA_Hybrid_Report.xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
        )
        if out_path:
            export_results_to_excel(self.results_data, out_path)
            messagebox.showinfo("Export Complete", f"Saved Excel QA Report to:\n{out_path}")


if __name__ == "__main__":
    app = OfflineAudioQAGUI()
    app.mainloop()
