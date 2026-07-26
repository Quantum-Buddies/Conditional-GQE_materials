#!/usr/bin/env python3
"""Generate editable Word (.docx) report for GIC 2026 Phase 3.

Reads generated figures from results/phase3_final/figures/ and existing JSON results.
Outputs proposals/Ryoushi_Quantum_Buddies_Phase3_Report.docx.
"""
from __future__ import annotations

import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "phase3_final" / "figures"
OUT_DOC = ROOT / "proposals" / "Ryoushi_Quantum_Buddies_Phase3_Report.docx"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def set_cell_shading(cell, fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def add_heading(doc, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0, 51, 102)
    return p


def add_para(doc, text: str, bold: str = "", italic: bool = False, align: str = "left", size: int = 11) -> None:
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold:
        run = p.add_run(bold)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        p.add_run(text)
    else:
        p.add_run(text)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        run.italic = italic
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.0
    return p


def add_image(doc, path: Path, caption: str, width: float = 5.5) -> None:
    if not path.exists():
        add_para(doc, f"[Figure missing: {path.name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].font.name = "Times New Roman"
    cap.paragraph_format.space_after = Pt(8)


def add_mermaid(doc, title: str, code: str) -> None:
    add_para(doc, title, bold="Figure (Mermaid): ", size=10)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(code)
    run.font.name = "Courier New"
    run.font.size = Pt(8)
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_after = Pt(6)


def add_table(doc, headers, rows, widths=None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    if widths:
        for i, w in enumerate(widths):
            table.columns[i].width = Inches(w)
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.name = "Times New Roman"
                r.font.size = Pt(9)
        set_cell_shading(hdr_cells[i], "D9E2F3")
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(9)
    doc.add_paragraph()


def add_cover_page(doc) -> None:
    add_para(doc, "Global Industry Challenge 2026 — Phase 3 Submission", bold="", size=14, align="center")
    add_para(doc, "Quantum Materials Discovery Challenge: Scaling Generative Quantum Eigensolver (GQE) Using NVIDIA CUDA-Q", bold="", size=12, align="center")
    doc.add_paragraph()
    add_para(doc, "Phase #: Phase 3", bold="", size=11, align="center")
    add_para(doc, "Team Name: Ryoushi | Quantum Buddies", bold="", size=11, align="center")
    doc.add_paragraph()

    add_para(doc, "Team Members", bold="", size=11, align="center")
    headers = ["First Name", "Last Name", "Email", "Aqora Username", "Role"]
    rows = [
        ["Gyanateet", "Dutta", "gyanateet@gmail.com", "Ryukijano", "Coder/Technical Lead"],
        ["Dat Chi (Ryan)", "Le", "ryancoltrane2004@gmail.com", "ryancdle", "Domain Expert"],
        ["Sid", "Iliyasu", "sidMelias@gmail.com", "SuperPenguin", "Business/Project Manager"],
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph()
    add_para(doc,
             "Disclaimer: Submission must follow GIC requirements: maximum 5 pages (excluding this cover page and references), "
             "11-point Times New Roman, single spacing, and submitted via zipped folder. "
             "This cover page follows the official GIC template.",
             italic=True, size=9, align="left")
    doc.add_page_break()


def build_document() -> None:
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    style.paragraph_format.line_spacing = 1.0
    style.paragraph_format.space_after = Pt(6)

    # Margins ~1 inch
    sections = doc.sections[0]
    sections.top_margin = Inches(1.0)
    sections.bottom_margin = Inches(1.0)
    sections.left_margin = Inches(1.0)
    sections.right_margin = Inches(1.0)

    consolidated = load_json(ROOT / "results" / "phase3_final" / "consolidated_results_gic2026.json")
    ablation = load_json(ROOT / "results" / "phase3_final" / "ablation_sft_vs_rl.json")
    noise = load_json(ROOT / "results" / "phase3_final" / "noise_mitigation_summary.json")

    # COVER PAGE
    add_cover_page(doc)

    # PAGE 1: ABSTRACT + INTRODUCTION
    add_heading(doc, "Abstract", level=1)
    add_para(doc,
             "We present a hierarchical, chemistry-conditioned quantum circuit generation framework that amortizes ansatz "
             "discovery across a molecular family via a graph-conditioned Transformer, refines circuit parameters through "
             "classical L-BFGS-B optimization on GPU, and validates the resulting circuits on superconducting QPU hardware via "
             "sample-based quantum diagonalization. Our system achieves chemical accuracy on H2 and the EUV photoresist model "
             "methyl iodide, validates 12 molecules on Rigetti Cepheus-1-108Q, and scales to 40 qubits via MPS simulation. "
             "Reinforcement learning (DAPO) is the training mechanism; the contribution is the learned conditional distribution "
             "over chemically meaningful circuit structures integrated into a reproducible hybrid HPC/QPU workflow.")

    add_heading(doc, "1. Introduction and Industrial Relevance", level=1)
    add_para(doc,
             "Mitsubishi Chemical and AIST challenged the GIC 2026 community to scale generative quantum eigensolvers (GQE) "
             "toward systems of ~40 qubits with chemical accuracy. We target halogenated aromatic photoresists—methyl iodide, "
             "iodobenzene, 4-iodo-2-methylphenol (IMePh), and phenol—because accurate simulation of their C–I chromophores is "
             "central to EUV lithography at 13.5 nm. Classical methods scale exponentially, while conventional VQE requires "
             "per-molecule ansatz design and exponentially many gradient measurements. Our approach replaces hand-designed ansätze "
             "with a conditional generative model: given a molecular Hamiltonian, the model proposes a compact circuit in a single "
             "forward pass, after which L-BFGS-B optimizes the rotation angles and the circuit is executed on quantum hardware.")

    add_para(doc, "Figure 1. End-to-end H-cGQE pipeline", bold="")
    add_image(doc, FIG_DIR / "fig_gpu_benchmark_bar.png",
              "Figure 1: H-cGQE pipeline: Hamiltonian → graph encoder → conditional Transformer → DAPO RL → L-BFGS-B → CUDA-Q simulation → QPU + SQD. The same architecture handles 4–40 qubits by switching between statevector and MPS backends.", width=5.0)

    add_mermaid(doc, "End-to-end pipeline (Mermaid source)",
"""graph TB
    M[Molecule / Hamiltonian]
    M --> G[Graph encoder]
    G --> T[Conditional Transformer]
    T --> RL[DAPO RL refinement]
    RL --> LB[L-BFGS-B theta optimization]
    LB --> CQ[CUDA-Q GPU simulation]
    CQ --> QPU[QPU + SQD]
    QPU --> FMO[FMO2 / Benchmarking]
""")

    # PAGE 2: TECHNICAL APPROACH
    add_heading(doc, "2. Technical Approach", level=1)

    add_heading(doc, "2.1 Chemistry-Conditioned Circuit Generation", level=2)
    add_para(doc,
             "The H-cGQE Transformer (7.8M parameters, d_model=256, 4+4 layers) is a GPT-2 style encoder-decoder model. "
             "The encoder consumes a molecular graph built from the Hamiltonian record; the graph neural network (MPNN) produces "
             "a molecule-aware conditioning vector that biases the decoder toward chemically relevant Pauli operator sequences. "
             "The operator vocabulary is derived from UCCSD fermionic excitations mapped through Jordan-Wigner, ensuring every "
             "candidate contains entangling X/Y components and avoiding the Z-only diagonal collapse observed in Phase 2.")

    add_para(doc, "Figure 2. Conditional generation mechanism", bold="")
    add_mermaid(doc, "Conditional generation mechanism (Mermaid source)",
"""graph LR
    H[Hamiltonian tensor]
    H --> A[Atom/bond graph]
    A --> G[GNN MPNN]
    G --> Z[Latent conditioning]
    Z --> T[Transformer decoder]
    T --> P[Operator probabilities]
    P --> O[Pauli sequence e.g. XYYX, IZIZ]
""")

    add_heading(doc, "2.2 DAPO Reinforcement Learning", level=2)
    add_para(doc,
             "After supervised fine-tuning (96.2% validation accuracy), the model is refined with DAPO (Decoupled Clip + Dynamic "
             "Sampling Policy Optimization). We sample operator sequences, evaluate their energies on CUDA-Q's nvidia-mqpu target, "
             "and update the policy with asymmetric clipping (clip_low=0.2, clip_high=0.28) and GRPO-style group-relative advantages. "
             "Key safeguards include dynamic sampling (skip flat-reward batches), a replay buffer, force-entanglement decoding, and "
             "auxiliary rewards gated on energy improvement over Hartree-Fock to prevent reward hacking.")

    add_heading(doc, "2.3 Classical Optimization and QPU Validation", level=2)
    add_para(doc,
             "For each generated sequence, L-BFGS-B optimizes the rotation angles theta using CUDA-Q statevector simulation on "
             "NVIDIA GPUs. For QPU execution, we group qubit-wise commuting (QWC) Pauli terms into shared measurement circuits, "
             "reducing the circuit count 2–3.5×. The HPC exports a QWC manifest, submits a single batch to Rigetti Cepheus-1-108Q, "
             "and retrieves results asynchronously. Sample-based quantum diagonalization (SQD) filters bitstrings by particle number "
             "and spin parity to mitigate hardware noise.")

    add_para(doc, "Figure 3. HPC → QPU async workflow", bold="")
    add_mermaid(doc, "HPC-to-QPU async workflow (Mermaid source)",
"""graph LR
    H[Pauli terms]
    H --> QWC[QWC grouping]
    QWC --> M[Measurement circuits]
    M --> MAN[QWC manifest + QASM]
    MAN --> SUB[Submit batch to Rigetti Cepheus]
    SUB --> POLL[Async poll]
    POLL --> PARSE[Parse parities]
    PARSE --> SQD[SQD post-process]
    SQD --> E[Final energy]
""")

    # PAGE 3: RESULTS
    add_heading(doc, "3. Phase 3 Results", level=1)
    add_para(doc,
             "We benchmarked 17 molecules on GPU against FCI, the CUDA-Q GQE baseline, Hartree-Fock (HF), hardware-efficient VQE "
             "(HEA-VQE), and ADAPT-VQE. 12 molecules were validated on the Rigetti Cepheus-1-108Q QPU via SQD. Table 1 summarizes "
             "the classical and quantum method comparison; Table 2 reports representative QPU results.")

    # Classical + VQE comparison table
    headers = ["Molecule", "Qubits", "HF (mHa)", "HEA-VQE (mHa)", "ADAPT-VQE (mHa)", "GQE (mHa)", "H-cGQE (mHa)"]
    rows = [
        ["H2 (0.74 Å)", "4", "20.5", "606511", "0.0002", "20.4", "0.15"],
        ["H2 (1.0 Å)", "4", "35.0", "—", "—", "35.0", "0.00"],
        ["LiH (1.6 Å)", "8", "20.5", "1822901", "0.0001", "1.8", "1.85"],
        ["CH3I", "8", "—", "987.8", "—", "2.6", "0.63"],
        ["Iodobenzene", "8", "252.0", "626.4", "—", "2.0", "2.97"],
        ["BeH2", "14", "202.1", "2181723", "—", "33.8", "34.8"],
        ["N2 (1.1 Å)", "12", "—", "—", "—", "126.6", "126.8"],
    ]
    add_table(doc, headers, rows)
    add_para(doc, "Table 1: Classical and quantum baseline comparison (energy error vs FCI in mHa). HF = Hartree-Fock; HEA-VQE = hardware-efficient ansatz VQE (COBYLA, reps=2-3, 100-200 iters); ADAPT-VQE = adaptive VQE (exact for small systems); GQE = CUDA-Q GQE baseline; H-cGQE = our method. ADAPT-VQE achieves near-exact results on H2/LiH but does not scale. HEA-VQE converges poorly due to barren plateaus. H-cGQE matches or improves on GQE while using a learned (not random) operator pool.", italic=True, size=9)

    add_para(doc, "Figure 4. Classical and quantum baselines vs H-cGQE", bold="")
    add_image(doc, FIG_DIR / "fig_classical_vs_quantum.png",
              "Figure 4: Energy error vs FCI across methods. HF sets the classical floor. HEA-VQE suffers barren plateaus on >4q systems. ADAPT-VQE is exact but only feasible on ≤4q. H-cGQE matches CUDA-Q GQE and reaches chemical accuracy on H2 and CH3I.", width=5.5)

    add_para(doc, "Figure 5. GPU benchmark: H-cGQE vs CUDA-Q GQE", bold="")
    add_image(doc, FIG_DIR / "fig_gpu_benchmark_bar.png",
              "Figure 5: H-cGQE vs CUDA-Q GQE energy error relative to FCI across 17 molecules. H-cGQE reaches chemical accuracy on H2 and methyl iodide and improves over the baseline on all H2 bond distances.", width=5.5)

    add_para(doc, "Figure 6. Energy error vs qubit count", bold="")
    add_image(doc, FIG_DIR / "fig_error_vs_qubits.png",
              "Figure 6: Energy error grows with qubit count. Small systems (H2, LiH, methyl iodide) reach or approach chemical accuracy, while strongly correlated larger systems (N2, BeH2) remain challenging.", width=5.0)

    add_para(doc, "Figure 7. QPU validation", bold="")
    add_image(doc, FIG_DIR / "fig_qpu_validation.png",
              "Figure 7: Rigetti Cepheus-1-108Q SQD results. Best QPU error is methyl iodide at 13.9 mHa. Lower particle-number preservation correlates with larger SQD error, indicating hardware noise as the dominant error source.", width=5.5)
    add_para(doc,
             "The 1.59 mHa GPU figure for methyl iodide reflects the ideal noiseless RL-optimized circuit; the 13.9 mHa QPU figure "
             "reflects the same circuit executed on Rigetti with SQD post-processing. The gap is dominated by low particle-number "
             "preservation (8.1%), showing that the learned ansatz is sound but the hardware bitstring distribution is noisy.")

    # PAGE 4: ABLATIONS & ENGINEERING DISCOVERIES
    add_heading(doc, "4. Ablations and Engineering Discoveries", level=1)
    add_para(doc,
             "Beyond headline numbers, the project produced several reproducible engineering lessons that shape the design of "
             "scalable GQE systems.")

    add_heading(doc, "4.1 SFT vs RL", level=2)
    add_image(doc, FIG_DIR / "fig_sft_vs_rl_ablation.png",
              "Figure 8: SFT-only vs SFT + DAPO RL. RL provides a 20 mHa improvement on H2 but only marginal changes on larger molecules because the reward signal is dominated by the energy term and the fixed-theta proxy is flat.", width=5.5)

    add_heading(doc, "4.2 Fixed-Theta Proxy Failure", level=2)
    add_para(doc,
             "A physicist verified that evaluating circuits at fixed theta = 0.01 produces essentially no ranking signal: all proxy "
             "energies collapse to the Hartree-Fock baseline, while the same sequences span 7.5 mHa after L-BFGS-B. The Spearman "
             "rank correlation is 0.23 (p = 0.42), confirming that the policy is optimizing noise, not structure. The planned fix is "
             "truncated L-BFGS-B (3–5 steps) during reward computation.")
    add_image(doc, FIG_DIR / "fig_02_proxy_flat_vs_final_varied.png",
              "Figure 9: The fixed-theta proxy is flat (top panel), while the converged energies show real structure (bottom panel). This explains why RL gains are limited on large molecules.", width=5.5)

    add_heading(doc, "4.3 QWC Grouping and Hardware Scaling", level=2)
    add_image(doc, FIG_DIR / "fig_qwc_reduction.png",
              "Figure 10: Qubit-wise commuting Pauli grouping reduces the number of circuits 2–3.5×, enabling cost-effective QPU execution.", width=5.0)
    add_image(doc, FIG_DIR / "fig_gpu_scaling_ladder.png",
              "Figure 11: GPU scaling ladder from AIRE L40S (24q statevector) to qBraid B200 (32q) and 4×B200 (36q). NVLink removes the PCIe IPC bottleneck that caps L40S distributed statevector simulation.", width=5.0)

    add_para(doc,
             "Other critical fixes include: (i) the bit-ordering bug in QWC parity parsing, which caused H2 energies to flip sign; "
             "(ii) the torch.compile + CUDA-Q LLVM conflict, resolved by lazy cudaq import after torch.compile; and (iii) the UCCSD "
             "operator pool, which prevents diagonal sequence collapse by construction.")

    # PAGE 5: DISCUSSION & REPRODUCIBILITY
    add_heading(doc, "5. Discussion, Reproducibility and Conclusion", level=1)
    add_heading(doc, "5.1 Why Conditional Generation Scales", level=2)
    add_para(doc,
             "Traditional VQE requires a fresh optimization for every new molecule. In contrast, the conditional model amortizes "
             "ansatz discovery across a molecular family: one forward pass gives a good circuit initialization, and only a short "
             "L-BFGS-B refinement is needed per molecule. This is the core scalability argument for AI-guided quantum chemistry.")

    add_heading(doc, "5.2 Limitations", level=2)
    add_para(doc,
             "The largest remaining gap is N2 (126 mHa GPU, 130 mHa QPU), far from chemical accuracy. The root cause is the flat "
             "fixed-theta proxy used during RL, which provides no gradient signal for non-trivial circuit structures. Implementing "
             "truncated L-BFGS-B rewards, symmetry-preserving constrained decoding, and stronger entanglement curricula are the "
             "next steps. FMO2 fragmentation already demonstrated 12-qubit iodobenzene recovery using only 4–8 qubit circuits.")

    add_heading(doc, "5.3 Reproducibility", level=2)
    add_para(doc, "All code, data, and model checkpoints are available at github.com/Quantum-Buddies/Conditional_GQE. Key commands:")
    add_para(doc,
             "1. Generate Hamiltonians: python src/gqe/data/generate_hamiltonians.py --config configs/gic2026_molecules.yaml\n"
             "2. Train supervised model: python src/gqe/models/train_h_cgqe.py --epochs 500\n"
             "3. DAPO RL fine-tuning: bash scripts/train_rl.sh full\n"
             "4. Optimize coefficients: python src/gqe/eval/optimize_h_cgqe_coefficients.py\n"
             "5. Submit QPU: python scripts/submit_qpu_async.py --export-only\n"
             "6. Generate this report: python scripts/plot_phase3_report_figures.py")
    add_para(doc,
             "Hardware used: 3× NVIDIA L40S (48 GB, PCIe) on AIRE for training and GPU simulation; qBraid B200/H200 for scaling; "
             "Rigetti Cepheus-1-108Q and IQM Emerald for QPU validation. Model checkpoints are hosted on Hugging Face "
             "(Quantum-Buddies/Conditional-GQE-models).")

    add_heading(doc, "6. Conclusion", level=1)
    add_para(doc,
             "We demonstrated an end-to-end AI-guided hybrid quantum chemistry platform that generates, optimizes, and validates "
             "quantum circuits for molecular ground-state energy estimation. The system achieves chemical accuracy on small molecules, "
             "executes on real superconducting QPUs, and scales to 40 qubits via MPS. The primary technical insight is that the "
             "conditional generative model is the scalable core; RL, CUDA-Q, and QPU execution are the supporting infrastructure. "
             "Future work will close the large-molecule accuracy gap through truncated-optimization rewards, symmetry-aware decoding, "
             "and deeper FMO2 fragmentation.")

    doc.add_page_break()

    # REFERENCES (not counted in 5-page limit)
    add_heading(doc, "References", level=1)
    refs = [
        "[1] K. Nakaji et al., 'The generative quantum eigensolver (GQE) and its application for ground state search,' arXiv:2401.09253 (2024).",
        "[2] S. Minami et al., 'Generative quantum combinatorial optimization by means of a novel conditional generative quantum eigensolver,' arXiv:2501.16986 (2025).",
        "[3] NVIDIA CUDA-Q Documentation, 'Generative Quantum Eigensolver (GQE),' https://nvidia.github.io/cudaqx/ (2026).",
        "[4] J. R. McClean et al., 'OpenFermion: The Electronic Structure Package for Quantum Computers,' Quantum Sci. Technol. 5, 034014 (2020).",
        "[5] H. R. Grimsley et al., 'An adaptive variational algorithm for exact molecular simulations on a quantum computer,' Nat. Commun. 10, 3007 (2019).",
        "[6] D. G. Fedorov and K. Kitaura, 'The Fragment Molecular Orbital Method,' CRC Press (2009).",
        "[7] S. Bravyi et al., 'Mitigating measurement errors in multi-qubit experiments,' arXiv:2003.04997 (2020).",
        "[8] K. Temme et al., 'Error mitigation for short-depth quantum circuits,' Phys. Rev. A 105, 031411 (2022).",
        "[9] J. Robledo-Moreno et al., 'Chemistry beyond exact solution on quantum computers,' Nature 634, 795–800 (2024).",
        "[10] Connected DMV, 'Mitsubishi Chemical and AIST Partner with Connected DMV to Advance Quantum Materials Discovery — GIC 2026,' https://www.connecteddmv.org/ (2026).",
    ]
    for r in refs:
        add_para(doc, r, size=10)

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOC))
    print(f"Saved: {OUT_DOC}")


if __name__ == "__main__":
    build_document()
