# Executive Summary  
We propose a **Hierarchical Conditional Generative Quantum Eigensolver (H‑cGQE)** approach for scalable EUV materials simulation.  In H‑cGQE, a transformer-based model is trained to map molecular Hamiltonians to quantum circuits (“H(x)→U”)【1†L42-L50】【19†L55-L60】.  We pre-train on small systems (e.g. H₂, LiH, BeH₂, N₂ dissociation) and fine-tune on halogenated photoresist fragments.  Unlike VQE, GQE places all parameters in the classical model, avoiding barren-plateau optimization【19†L125-L131】.  This enables **zero-shot generalization**: once trained, the model can generate near-optimal circuits for new Hamiltonians without retraining【1†L42-L50】【19†L55-L60】.  We then introduce a **hierarchical scaling strategy**: large molecules are decomposed into active-space fragments, each solved by c-GQE, and their solutions recombined via a many-body expansion (e.g. FMO/EFMO) to recover the full energy.  NVIDIA’s CUDA-Q platform provides GPU/MPI acceleration, with demonstrated speedups of ~40× on one H100 and ~8× across eight GPUs【19†L60-L63】【13†L508-L515】.  We will build a prototype in this stack and report benchmarks (energy errors ΔE, circuit depth, 2‑qubit gate count, QPU calls) comparing H‑cGQE vs VQE baselines.  

# Phase 2 Deliverables and Compliance  
- **Submission:** A 3-page PDF (plus cover page and references) in 11‑point Times New Roman, single-spaced【28†L292-L300】.  We will use the official GIC cover template with all team members listed. The file will be named `TeamName__Phase2_Version1.pdf`【28†L292-L300】.  
- **Executive Summary and Content:** The report will address the use case focus, technical approach, data/model strategy, and computing resources. We will include mermaid diagrams for architecture, timeline, and fragmentation. Key figures (energy-error plots, generalization curves, scaling projections) and tables (baseline vs proposed metrics) will be provided.  
- **Code Release:** A reproducible code repository (e.g. GitHub) with scripts for Hamiltonian generation, transformer training, and evaluation. The repository will include instructions and environment specs (assume access to a single-node machine with **4 GPUs (48–96 GB each)** and sufficient CPU/RAM).  

# Technical Approach: H‑cGQE Architecture

## Hamiltonians & Dataset  
We curate molecular Hamiltonians in STO‑3G basis and specify active spaces:  
- **Training set:** H₂ (2 electrons, 4 spin-orbitals → 4 qubits), LiH (4 e, ~6 spin‑orbitals), BeH₂ (6 e, ~10 spin‑orbitals), N₂ (10 e in valence, ~16 spin‑orbitals) along their bond‑dissociation curves. All use STO‑3G; active spaces are chosen to capture valence (e.g. freeze core 1s orbitals).  
- **Fine‑tune set:** Halogenated aromatics (C₆H₅I iodobenzene, and 4-iodo-2-methylphenol C₇H₆IO). We apply effective core potentials (ECPs) for I, keeping e.g. 4 valence e⁻ and 4 orbitals → ~8 qubits each.  
- **Scaling proxy:** A tin-oxo cluster (Sn₄O₄(OH)₄) truncated model with relativistic ECPs on Sn. We select a small active space (e.g. 8 e⁻, 8–12 orbitals, ~16–24 qubits) as a demonstration of fragmentation strategy (see below).  

For each molecule we use PySCF/OpenFermion to obtain the Pauli Hamiltonian【15†L78-L85】.  Example (PySCF+OpenFermion):  
```python
from pyscf import gto
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner

mol = gto.Mole(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', charge=0, spin=0).build()
mf = run_pyscf(mol, run_scf=1, run_fci=1)
fermion_ham = mf.get_molecular_hamiltonian()  # OpenFermion FermionOperator
qubit_ham = jordan_wigner(fermion_ham)       # OpenFermion QubitOperator
```
This yields the full qubit Hamiltonian (as a sum of Pauli strings).  We store coefficient vectors as model inputs.  

## Transformer Model & Tokenization  
We use an encoder–decoder transformer (GPT‑2 style) that conditions on the Hamiltonian embedding.  The **encoder** takes Pauli coefficients (flattened vector) plus optional graph/GNN embedding of molecular structure. The **decoder** is an autoregressive GPT generating a sequence of discrete “gate tokens” (e.g. *“X0”, “Y1”, “CNOT2_3”, etc.*) from a predefined gate vocabulary.  The vocabulary will include physically motivated operations (single-qubit rotations and two-qubit entanglers from a UCCSD-like pool【19†L150-L156】).  Each generated token corresponds to one unitary gate (on one or two qubits).  

Training is fully supervised (no RL).  We generate target circuits by:  
- **Exact-circuit targets:** For small systems (H₂, LiH, etc.), we obtain exact ground states from FCI and use a circuit synthesis algorithm (e.g. Qiskit’s circuit drawer or customized Ansatz) to derive a minimal circuit that prepares the FCI state.  
- **VQE-derived targets:** We run ADAPT-VQE or UCCSD-VQE (using Qiskit or Cirq) for each training Hamiltonian to get a near-optimal ansatz. These circuits serve as additional labels. (ADAPT-VQE iteratively builds an ansatz circuit to reach FCI accuracy【17†L5-L11】.)  

The loss function is cross-entropy on token prediction plus a performance loss: we also sample generated circuits during training and backpropagate the energy expectation value (via backprop through soft sampling or REINFORCE-like surrogate) to bias the model toward low-energy circuits【19†L139-L146】.  We pre-train the transformer on the small-molecule Hamiltonians (covering a range of geometries) and then **fine-tune** on the iodinated fragment data.  This parallels Minami *et al.*’s demonstration of pretraining a conditional-GQE on one domain and fine-tuning on related problems【1†L42-L50】【19†L139-L146】.  

## Hierarchical Fragmentation and Scaling  
To tackle larger systems, we decompose the molecule’s orbital space into fragments (active spaces) and solve each with c-GQE independently.  We employ **active-space selection** to partition orbitals into chemically-meaningful subsets (e.g. bonding/antibonding pairs, or EFMO monomers/dimers【21†L27-L35】).  Each fragment Hamiltonian Hᵢ is fed to the (same) conditional-GQE model to produce a fragment circuit Uᵢ, so that overall circuit U≈∏ᵢUᵢ (with possible ordering or layering).  

For example, in an FMO-like many-body expansion, total energy = ∑ᵢE[Uᵢ] – ∑_{i<j}E[Uᵢ+Uⱼ] + ….  We plan to implement at least monomer+dimer recombination: run c-GQE on each fragment and on every fragment pair, then combine energies via inclusion–exclusion (analogous to spatial fragmentation【21†L27-L35】).  Alternatively, perturbative corrections (MP2-like) can refine energies across fragments. This hierarchical approach can **reduce qubit requirements by ~50%** or more【21†L27-L35】: fragments can be solved on 8–12 qubits even if the full molecule would need many more.  

```mermaid
flowchart LR
  H[Molecular Hamiltonian H(x)] --> Split{Fragment Decomposition}
  Split --> |fragment H1| cGQE1[Transformer (c-GQE) → U1]
  Split --> |fragment H2| cGQE2[Transformer (c-GQE) → U2]
  cGQE1 --> Circuit1[Fragment Circuit U1]
  cGQE2 --> Circuit2[Fragment Circuit U2]
  Circuit1 --> Compose[Compose U1·U2 ...]
  Circuit2 --> Compose
  Compose --> QPE[Quantum Execution (Energy + q-sc-EOM)]
  Compose --> Combine[Energy Recombination via Many-Body Exp.]
```

Each fragment circuit is shallow (few layers) because GQE generates efficient ansätze【19†L139-L146】.  Since evaluation of fragments can run in parallel on multiple GPUs/CPUs, the overall scheme scales well.  NVIDIA’s CUDA-Q already supports distributed GQE: “The algorithm can efficiently utilize multiple QPUs through MPI for parallel operator evaluation, making it suitable for larger quantum systems”【13†L508-L515】.  We will leverage CUDA-Q or Qiskit Aer with multi-GPU backends.  A complexity analysis will be provided: for *m* fragments of *k* qubits, training cost ~O(m·poly(k)), and inference cost is linear in the number of generated circuits.  In practice, GPUs give massive speedup: NVIDIA reports ~40× speedup on one H100 vs CPU, and ~8× further on an 8‑GPU node【19†L60-L63】.  

## Implementation Plan and Timeline  
We assume access to a single-node machine (e.g. NVIDIA DGX) with 4 GPUs (48–96 GB each) and sufficient CPU/memory.  We allocate tasks as follows:

```mermaid
gantt
    title Project Timeline to May 31, 2026
    dateFormat  YYYY-MM-DD
    axisFormat  %m-%d
    section Preparation
    Hamiltonian Dataset Preparation  :a1, 2026-05-07, 4d
    Transformer Model Setup         :a2, after a1, 2d
    Circuit Pool and Token Vocabulary: a3, after a2, 2d
    section Training & Evaluation
    Pretrain on Small Molecules      :b1, after a3, 5d
    Evaluate Zero-Shot (Holdouts)    :b2, after b1, 3d
    Fine-Tune on Iodinated Fragments :b3, after b2, 4d
    Fragmentation Implementation    :b4, after b3, 4d
    section Analysis
    Baseline (VQE) Runs             :c1, parallel b1, 5d
    Collect Results (Plots/Tables)  :c2, after b4, 3d
    Write Report & Figures          :c3, 2026-05-25, 3d
    Final Revisions & Submission    :c4, 2026-05-29, 2d
```

Key milestones:  
- **By May 15:** Complete pretraining on H₂/LiH/BeH₂/N₂ and test generalization.  
- **By May 20:** Finish fine-tuning on iodinated molecules and implement fragment-energy recombination.  
- **By May 25:** Run VQE baselines (variations: UCCSD, ADAPT-VQE) and collect metrics.  
- **By May 28:** Generate final plots (energy error vs molecule, generalization curves, scaling extrapolations).  
- **By May 30:** Finalize PDF (format check) and code release.  

We estimate compute: PySCF Hamiltonian runs (<12 qubit) are trivial (<minutes). Transformer training (millions of parameters) will use all 4 GPUs with PyTorch. Pretraining might take ~1 day; fine-tuning ~0.5 day. VQE baselines (Qiskit) use CPU/GPU; expect <2 days total with parallel.  

# Performance Metrics and Baselines  
We will report:  
- **Energy error (ΔE)** relative to exact diagonalization / FCI for each test Hamiltonian.  
- **Circuit depth and two-qubit gate count** of the generated ansatz.  
- **Quantum hardware calls**: number of circuit evaluations or shots needed.  

Tables will compare H-cGQE vs:  
1. Standard VQE with random initial parameters  
2. VQE with Hartree-Fock initial guess  
3. ADAPT‑VQE circuits  
Each baseline will be run for comparable shot budgets.  From Keithley *et al.*, GQE circuits used ~50% fewer gates than VQE for water【9†L75-L77】; we expect similar or better for our targets.  

```markdown
| System       | Qubits | Method       | ΔE (mHa) | CNOTs | Depth | QPU calls |
|--------------|--------|--------------|----------|-------|-------|-----------|
| H₂          | 4      | VQE (UCCSD)  | ...      | ...   | ...   | ...       |
| H₂          | 4      | c-GQE        | ...      | ...   | ...   | ...       |
| LiH         | 6      | VQE (UCCSD)  | ...      | ...   | ...   | ...       |
| LiH         | 6      | c-GQE        | ...      | ...   | ...   | ...       |
| BeH₂        | 10     | VQE (UCCSD)  | ...      | ...   | ...   | ...       |
| BeH₂        | 10     | c-GQE        | ...      | ...   | ...   | ...       |
| Iodobenzene | ~8     | VQE (ADAPT)  | ...      | ...   | ...   | ...       |
| Iodobenzene | ~8     | c-GQE        | ...      | ...   | ...   | ...       |
```

Figures will include: (a) **Energy error vs. molecule/geometry** (showing c-GQE matching exact vs VQE errors), (b) **Generalization curves** (train vs test performance as function of number of qubits or Hamiltonian complexity), and (c) **Scaling projection** (e.g., circuit evaluation time vs qubit count, based on CUDA-Q speedups【19†L60-L63】).  

# Mermaid Diagrams  
```mermaid
flowchart TD
    Input[Hamiltonian H(x)] --> Encode[Pauli Embedding + GNN]
    Encode --> Transformer[Encoder–Decoder Transformer]
    Transformer --> Output[Circuit U = (g₁, g₂, …)]
    Output --> QuantumExec[Quantum Execution (measure E₀, q-sc-EOM)]
```
```mermaid
flowchart LR
    Mol[Large Molecule H] --> FragmentA[Fragment A Active Space]
    Mol --> FragmentB[Fragment B Active Space]
    FragmentA --> cGQE_A[Conditional GQE → U_A]
    FragmentB --> cGQE_B[Conditional GQE → U_B]
    cGQE_A --> Combine[Compose U_A·U_B]
    cGQE_B --> Combine
```
```mermaid
gantt
    title Implementation Timeline
    dateFormat  YYYY-MM-DD
    section Prep
    DFT/Hamiltonians: done            :a, 2026-05-01, 1w
    Circuit Targets: done             :b, 2026-05-01, 1w
    section Development
    cGQE Pre-training                :c, after a, 10d
    Baseline VQE Runs               :d, parallel c, 8d
    Fine-tuning on Halides         :e, after c, 7d
    Fragmentation Workflow         :f, after e, 6d
    section Finalization
    Analysis & Plotting            :g, after f, 4d
    Draft Report                   :h, after g, 3d
    Submit PDF/Code                :i, 2026-05-30, 1d
```

# Checklists and Assumptions  
- **Assumptions:** Use of 4× GPUs (48–96 GB each) on CUDA-Q or qBraid.  No reinforcement learning (all supervised).  Expect ~10^4–10^5 training examples (circuit samples).  Compute budget not given; assume we optimize for ~1–2k training epochs on small data.  
- **Prioritized Deliverables:**  
  - [ ] *3-page PDF* (excluding cover/references) in correct format【28†L292-L300】  
  - [ ] **Figures:** energy-error plot, generalization curve, scaling chart  
  - [ ] **Tables:** baseline vs c-GQE metrics (ΔE, CNOT count, depth, evaluations)  
  - [ ] **Mermaid diagrams** in appendix or main doc for architecture, timeline, fragmentation  
  - [ ] **Code repository** with: Hamiltonian generation (PySCF/OpenFermion), transformer training (HuggingFace/PyTorch code), VQE baselines (Qiskit scripts)  
  - [ ] **Environment specification:** assume single-node multi-GPU (e.g. PyTorch with CUDA, PySCF, OpenFermion, Qiskit).  
  - [ ] **Citations:** e.g. conditional‑GQE【1†L42-L50】, GPT-QE【7†L63-L68】, Auger GQE【9†L69-L77】, NVIDIA scaling【19†L60-L63】, OpenFermion usage【15†L78-L85】, fragmentation【21†L27-L35】.  

# References  
Minami *et al.* (2025) conditional-GQE【1†L42-L50】; Nakaji *et al.* (2024) GQE/GPT-QE【7†L63-L68】; Keithley *et al.* (2026) GQE for spectra【9†L69-L77】; NVIDIA CUDA-Q GQE docs【13†L508-L515】; NVIDIA blog【19†L55-L63】【19†L125-L131】; OpenFermion/PySCF tutorials【15†L78-L85】; Fragmentation methods【21†L27-L35】.