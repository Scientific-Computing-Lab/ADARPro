# Post-hoc checks on the selected predictor's canonical-test predictions

These files support the three post-hoc clauses in SI Appendix, section L ("post-hoc checks placed it in ranking
duplexes rather than adenosines within a duplex, found it carried by a minority of pairs, and found that in Combined
its Brier improvement depended on the bootstrap seed"). All three checks were made after the selected predictor's
canonical-test session, on its saved predictions. They are descriptive: no model, weight, threshold or choice was
changed by them.

Names used in the files: `flagship` = the selected predictor; `equal_average` = the equal-weight average of the same
two networks; `pnas` = the first-round predictor; `adaredit` = reproduced supplied-checkpoint AdarEdit.

| Clause | Output | Producer |
|---|---|---|
| The gain sits in ranking duplexes, not adenosines within a duplex (Brier split into refinement and calibration; F1 at the frozen threshold vs the best threshold; gap-term ablation of the frozen stage) | `s06_ranking_vs_calibration.csv`, `.txt` | `scripts/C_recompute/s06_ranking_vs_calibration.py` |
| Carried by a minority of pairs (AUROC difference after dropping the k pairs that favour the selected predictor most, k = 1, 3, 5, 10, 20, against each comparator) | `06_drop_topk.csv` | see note below |
| In Combined, the Brier improvement over the equal-weight average depends on the bootstrap seed (40 seeds x 2,000 pair resamples) | `s03c_borderline_seeds.csv`, `.txt` | `scripts/C_recompute/s03c_borderline_seeds.py` (helpers in `common.py`) |

Inputs: the per-row canonical-test predictions of that session, which are not deposited (only aggregate test
results are; `results/adarpro/final_locked_test_v3/`). The scripts therefore document the computation; they cannot
be rerun from a public clone. They are copies of the scripts as run, with absolute local paths replaced by
`<repository>`, `<this folder>` and `<output folder>`; nothing else was changed.

Note on `06_drop_topk.csv`: it was written by a larger script that also runs checks not reported in the paper; that
script is kept in the authors' records rather than published. The method: each pair's influence on the AUROC difference was measured by leaving it out
(influence = the difference on all rows minus the difference without that pair); pairs were ordered by influence, the
k pairs that raise the difference most were removed, and the AUROC difference was recomputed on the remaining rows
(`d_auroc`; `full` is the difference on all rows).
