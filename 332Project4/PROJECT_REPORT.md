# Online Market Design Project — Parts 1 and 2 (Full Walkthrough)

Contents
- Part 1: Online Reserve Pricing (single‑item second‑price with reserve)
- Part 2: Position Auctions (VCG, Myerson, and an online plug‑in learner)
- Amazon‑inspired Extension (quality scores and per‑slot dynamic reserves)
- Experimental design, figures, and takeaways
- Reproducibility and what to run
- Mapping to the assignment requirements (by item)
- Notes on class content and why these choices are faithful

---

## Part 1 — Online Reserve Pricing (Single‑Item, Second‑Price with Reserve)

Goal (from `project_description.md`): Learn an optimal reserve price online in a truthful second‑price auction, with i.i.d. bidders whose values are drawn from a distribution F. Compare the learner’s performance to the optimal revenue for the known distribution. Explore variations: different F, different numbers of bidders, and additionally multi‑unit (k items sold via (k+1)‑st price).

Why this setting
- A second‑price auction with reserve is a canonical truthful mechanism for a single item. Truthful bidding lets us focus on revenue optimization via the reserve rather than strategic reporting.
- This directly reflects class content: optimal auctions (Myerson), reserve pricing, and online learning applied to mechanism design.

What we built
1) Two value distributions F:
   - Uniform[0,1] (textbook baseline).
   - Quadratic CDF F(z) = z^2 on [0,1] (non‑standard but easy to sample via inverse transform z = √q) - got this from project desc.
2) Bidders per round: m ∈ {2,5,10}, truthful bids = values.
3) Revenue rule per round:
   - If max bid < r: no sale, revenue = 0.
   - Else: winner pays max(r, second‑highest bid).
4) Two learning approaches:
   - Bandit baseline over a discretized reserve grid (epsilon‑greedy, εt = min{1, c/√t}). Rationale: dead‑simple online benchmark that does not exploit structure.
   - Myerson plug‑in: estimate the CDF F empirically from past bids; choose reserve r that maximizes z(1 − F̂(z)) on a fine grid. Rationale: in a truthful, i.i.d. setting, the monopoly price is structure we can exploit to learn faster.
5) Measurement and comparisons:
   - Average revenue vs time (running mean).
   - Confidence intervals (multi‑seed runs) and cumulative regret to the optimal benchmark.
   - For Uniform, m=2, we verify the analytic benchmark r* = 1/2 with expected revenue 5/12 ≈ 0.4167.
6) Variations and extensions:
   - Different m and F; the plug‑in should benefit from larger m (more informative rounds).
   - Optional multi‑unit extension: sell k items using the truthful (k+1)‑st price with a reserve; we implement the revenue function and a quick demo (e.g., k=4).

Why these algorithms
- Bandit baseline (discretized reserves): aligns with the assignment’s suggestion to “simply apply an online learning algorithm.” It is transparent and produces interpretable regret/revenue curves.
- Myerson plug‑in: maps course theory (virtual values/monopoly prices) to a data‑driven, truthful learner. It uses structure beyond bandits and, as expected in class, typically converges faster and more stably.

Key results (qualitative)
- Both learners converge towards the optimal fixed reserve’s expected revenue.
- The plug‑in generally reaches that level faster and with lower early regret.
- Increasing m tends to increase average revenue and speed stabilization (more competition and more informative observations).
- The quadratic distribution shifts the optimal reserve; convergence trends remain similar.
- With expected‑clicks measurements and fixed RNG seeds, the curves are smooth and the performance gap is clear.

Figures referenced
- figures/rev_uniform.png, figures/res_uniform.png
- figures/rev_quadratic.png, (optional) figures/res_quadratic.png
- figures/rev_uniform_ci.png, figures/regret_uniform_ci.png
- figures/rev_quadratic_ci.png, figures/regret_quadratic_ci.png
- (Optional extension) figures/rev_uniform_k4.png

How this addresses the Part 1 requirements
- Truthfulness: second‑price with reserve is truthful; bidders bid values.
- Online learning: bandit baseline and Myerson plug‑in.
- Comparisons to optimal revenue: analytic check for Uniform,m=2; Monte Carlo optimal reserve/revenue for all other settings.
- Variations: different F, different m, and multi‑unit option.
- Convergence analysis: revenue vs. time, confidence intervals across seeds, cumulative regret.

---

## Part 2 — Position Auctions (Truthful Mechanisms + Online Learning)

Prompt 3 (from `project_description.md`): In a position auction with positions 1…m and click probabilities w1 ≥ … ≥ wm, identify the optimal truthful mechanism (given value distributions F1,…,Fm). Then design an online learning algorithm to learn this mechanism. Compare online performance to theoretical baselines.

Why this setting
- Position auctions with separable click probabilities are the standard model for sponsored search and ad positions (covered in class). They are the natural multi‑slot analogue of Part 1.
- This allows clean connections to VCG (welfare) and to Myerson’s revenue‑optimal mechanism with position weights.

Modeling assumptions
- Values are per‑click. Expected clicks in slot j occur with probability wj. We compute expected revenue per round (no Bernoulli click noise) to isolate mechanism/learning effects and make convergence diagnostics clearer.
- Independent private values vi ~ Fi (homogeneous options: uniform or quadratic for simplicity).

Truthful baselines implemented
1) VCG (welfare‑optimal, truthful):
   - Rank by values and allocate top bidders to top slots (w1 ≥ … ≥ wm).
   - Payments equal the externality on others per expected clicks.
   - VCG is the right welfare baseline (course content).
2) Myerson (revenue‑optimal, truthful):
   - Compute virtual values φi(vi) = vi − (1 − Fi(vi))/fi(vi).
   - Allocate to maximize Σj wj · φi(vi) (iron if needed); payments via Myerson identity.
   - We use closed forms for Uniform and for the quadratic CDF to build an oracle revenue curve.

Online algorithm
- Myerson plug‑in learner for position auctions:
  - Maintain an empirical CDF (and a smoothed density) for each bidder to estimate φ̂i each round.
  - Allocate to maximize Σj wj · φ̂i(vi); compute payments from the same structural rationale.
  - Rationale: directly targets the truthful revenue‑optimal allocation rule, rather than treating the allocation/pricing space as a black box. In line with class lectures, it should converge faster than unstructured methods.

What we compare
- Average revenue vs time: oracle Myerson (theoretical best truthful revenue), VCG (welfare, truthful), and the plug‑in learner.
- CI and regret: Multi‑seed runs for 95% confidence bands; cumulative regret vs the oracle quantifies learning speed.
- Sweeps in m and w: check that more bidders or stronger top slots increase revenue and speed stabilization.

Key results (qualitative)
- The plug‑in converges to the oracle’s revenue level (top line). VCG is lower (as expected for welfare vs revenue).
- Quadratic distribution tends to yield higher revenues than Uniform (values concentrate towards higher ranges).
- CI bands shrink as T grows; cumulative regret grows sublinearly; plug‑in usually exhibits lower early regret than VCG.
- Sweeps confirm comparative statics: higher m and larger w1,w2,... raise revenue and speed stabilization.

Figures referenced
- figures/part2/rev_uniform.png (revenue vs time for uniform)
- figures/part2/rev_quadratic.png
- figures/part2/rev_uniform_ci.png, figures/part2/regret_uniform_ci.png
- figures/part2/rev_quadratic_ci.png, figures/part2/regret_quadratic_ci.png
- figures/part2/sweep_m_uniform.png (example sweep)

Why this solves Prompt 3
- We identify the optimal truthful mechanism (Myerson with position weights) and the welfare‑optimal VCG.
- We implement an online learner that learns the Myerson mechanism from data (plug‑in via empirical CDFs).
- We quantify how closely online performance matches theory (revenue, CI, regret) and test comparative statics.

---

## Amazon‑Inspired Extension (Practice Layer)

Motivation
- The Amazon Ads document describes real‑world auction layers: quality/relevance signals and (dynamic) reserve logic tailored to placements and contexts. While the core auction can be VCG or GSP‑like, practical systems often use per‑slot thresholds and quality adjustments.

What we added
- Quality scores qi per bidder (relevance). Practice baseline: GSP with score si = qi · bi (we set bi = vi to focus on pricing/allocation while keeping bids truthful for comparison).
- Per‑slot dynamic reserves rj(t): learned online using a simple epsilon‑greedy bandit per slot.
- We compare GSP(q) without reserves vs GSP(q) with learned per‑slot reserves (extension only; this is not a truthful mechanism, but it reflects practice like the Amazon notes).

Observations
- Quality changes ranking (and payments in GSP). The impact on revenue depends on qi and the w profile.
- Learned slot reserves stabilize over time; they can increase revenue relative to the no‑reserve baseline in many settings.
- This mirrors the rationale for “reserve pricing functions” and relevance‑driven ranking in platform reality.

Figure
- figures/part2/ext_quality_reserves.png

How this relates to class content
- While GSP is not truthful in general, the extension is explicitly flagged as “practice‑like.” The core truthful analysis relies on VCG and Myerson in the main Part 2. The extension connects theory to realistic system knobs (quality and per‑slot thresholds).

---

## Experimental Design and Implementation Notes

Common to both parts
- We use expected‑outcome revenue (expected price in Part 1; expected clicks × price per click in Part 2) to reduce noise and keep convergence signals clear.
- Multi‑seed runs (typically 3) provide 95% CIs around mean revenue and cumulative regret.
- Bandit exploration uses εt = min{1, c/√t}, c ≈ 1 by default.
- Grids: 101 points for bandit reserve grid; 1001 points for plug‑in’s 1D maximizations (reserve argmax or virtual‑value grids).
- All figures are written to `figures/` (Part 1) and `figures/part2/` (Part 2).

Part 1 specifics
- Reserve optimization over [0,1] for the value support.
- Bandit mean‑reward tracking per arm; plug‑in uses empirical CDF + mild density smoothing to form F̂ and f̂.
- Optional k‑unit revenue for (k+1)‑st price with a reserve (demo run provided).

Part 2 specifics
- VCG implementation carefully handles K = min(m, #slots) and expected‑click payment decomposition.
- Oracle Myerson: closed‑form virtual values for Uniform and F(z)=z^2; allocation by virtual values times w.
- Plug‑in Myerson: empirical CDF and smoothed density per bidder → φ̂i; allocate by Σ wj φ̂i.
- CI/regret functions mirror those used in Part 1.
- Sweeps produce tables/plots showing how m and w alter final average revenue.
- Extension runner returns both revenue series (with/without reserves) and per‑slot reserve trajectories.

---

## Reproducibility — What to Run

Environment
- The project ships with notebooks under `notebooks/`. Figures are saved into `figures/` (Part 1) and `figures/part2/` (Part 2).

Part 1
1) Open `notebooks/part1_reserve_learning.ipynb`.
2) Run the setup and baseline/plug‑in cells.
3) Basic runs: set `RESULTS = run_all_experiments()` and then call the plotting helpers:
   - `plot_revenue_curves(RESULTS["uniform"], ...)`
   - `plot_reserve_trajectories(RESULTS["uniform"], ...)`
4) Optional: run the CI/regret cells and the multi‑unit demo.
5) Export summary row(s) via the provided exporter (CSV under `results/`).

Part 2
1) Open `notebooks/part2_position_auctions.ipynb`.
2) Run the setup and theory/baseline cells (VCG and oracle Myerson).
3) Run the plug‑in learner and the “example experiments and plot” cell to generate `figures/part2/rev_{dist}.png`.
4) CI/regret: use `run_rounds_multiseed(...)` and `plot_ci_curves(...)` / `plot_regret_ci(...)`.
5) Sweeps: use `sweep_m_w(...)` and `plot_sweep_table(...)` to confirm trends over m and w.
6) Extension: define a quality vector q and run `run_quality_extension(...)`; then `plot_quality_extension(...)`.
7) Export summary rows with `export_summary_row(...)` (CSV to `results/part2_summary.csv`).

---

## Mapping to Assignment Requirements (Checklist)

- Part 1:
  - Learn reserve online in a truthful second‑price auction ✔ (bandit + plug‑in).
  - Compare to optimal revenue ✔ (analytic for U[0,1],m=2; Monte Carlo otherwise).
  - Variations in F and m ✔ (Uniform, quadratic; m ∈ {2,5,10}).
  - Multi‑unit option ✔ (k‑unit (k+1)‑st price demo).
  - Convergence and analysis ✔ (revenue vs time, CI, regret, reserve trajectories).
- Part 2:
  - Identify optimal truthful mechanism for position auctions ✔ (Myerson + VCG).
  - Online algorithm to learn it ✔ (Myerson plug‑in with empirical CDFs).
  - Compare to theory ✔ (oracle Myerson, CI/regret, sweeps).
  - Connect to practice ✔ (extension with quality scores + slot‑level reserves).

---

## Notes on Class Content and Justification

- Truthful mechanisms: We used textbook VCG (welfare) and Myerson (revenue) as ground truth in Part 2; second‑price with a reserve is truthful in Part 1. This matches lecture emphasis on dominant strategies and virtual values.
- Myerson structure: The plug‑in approach is the standard way to operationalize Myerson’s mechanism from data—estimate the CDF/density and substitute into the virtual‑value expression. This is exactly the “structure exploitation” you see in class when comparing generic learning vs learning guided by mechanism theory.
- Online learning: Epsilon‑greedy bandits (Part 1 baseline) illustrate model‑free selection on a 1D action grid. The plug‑in in both parts shows faster convergence when theory is leveraged.
- Practice extensions: Quality scores and per‑slot reserves mirror real ad platforms. These are not theoretical requirements for truthfulness but are critical in practice for ranking and monetization—aligning with the Amazon Ads notes.

---

## Conclusive Takeaways

- Part 1: Both the bandit and the Myerson plug‑in converge to the optimal reserve’s expected revenue; the plug‑in is faster and more stable. More bidders increase expected revenue and speed stabilization. The quadratic distribution shifts the optimal reserve and raises revenue (as values skew higher). Optional k‑unit results behave analogously.
- Part 2: The plug‑in learner approaches the oracle Myerson revenue; VCG trails (welfare vs revenue). CI bands shrink and regret grows sublinearly, indicating efficient learning. Increasing m and strengthening top click probabilities raise revenue and speed stabilization. The extension with quality scores and per‑slot reserves demonstrates practice‑like knobs that can lift revenue and stabilize policy, consistent with how platforms manage relevance and thresholds.
- Overall: The project applies course theory (truthfulness, virtual values, VCG) to online learning, demonstrating how structure‑aware learners outperform generic baselines and how practical layers (quality, reserves) can be layered on top of theoretical mechanisms in real systems.

---

## Appendix: File/Artifact Map

- Part 1 notebook: `notebooks/part1_reserve_learning.ipynb`
  - Bandit learner; Myerson plug‑in; CI/regret; multi‑unit demo; summary CSV exporter
- Part 2 notebook: `notebooks/part2_position_auctions.ipynb`
  - VCG and Myerson (oracle); plug‑in learner; CI/regret; m and w sweeps; extension; summary CSV exporter
- Figures:
  - Part 1: `figures/*.png`
  - Part 2: `figures/part2/*.png`
- Summaries: `results/summary.csv`, `results/part2_summary.csv`
- Slides: `presentation.tex` (with extensive collaborator notes)


