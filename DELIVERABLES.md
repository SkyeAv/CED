# CED paper deliverables

Status: focused implementation and analysis plan, not generated results.

Target journal: MDPI Journal of Personalized Medicine (JPM).

Working title: **Graph neural representation learning for family stratification and treatment hypothesis generation in Camurati-Engelmann disease**.

## 1. Paper questions and scope

1. How do all 24 CED type 1 families group in the graph autoencoder's hidden embedding space?
2. Which clinical features, TGFB1 variants, and graph evidence distinguish those groups?
3. Which known and novel drugs receive high predicted treatment-association scores for CED overall and for each family?

The main outputs are a family subtype atlas/datamap, treatment prediction tables, and interpretable evidence-backed explanations. Standardized future-patient matching is a supplementary research prototype, not a second major modeling project.

Use all 24 families. The small cohort is appropriate for an exploratory rare-disease study, with transparent uncertainty and sensitivity analyses. Call groups **candidate family subtypes** or **family clusters**, not established clinical subtypes. An unstable partition is still reportable as exploratory grouping, not a validated taxonomy.

## 2. Preserve the graph and fix analysis correctness

Keep the current canonical graph, including `applied_to_treat`, concrete `treats` predicates where genuinely present, family qualifiers, provenance, and external drug-target/pathway evidence. Do not replace treatment exposure with efficacy assertions, merge distinct predicates, or invent family-treatment triples.

Required before paper runs:

- Connect real categorical, numeric, and semantic text features to training. `src/ced/model/training.py` currently constructs a `RandomProjectionProvider`; random features remain a test/control provider, not the paper's semantic model.
- Export the encoder hidden state as `z_node [N,D]`, before the ComplEx projection, with exact entity IDs. The current model constructs `D=32`; read the dimension from the checkpoint rather than assuming 64.
- Audit family phenotype proportions currently annotated as `effect_type: odds_ratio`. Expose verified values as `affected_fraction`, not odds ratios or treatment effects.
- Preserve reported positive fractions, observed zero, and missing separately. Observed-zero `has_phenotype` rows must not become positive phenotype-presence supervision merely because a statement exists.
- Retain verified denominators and provenance. Reconstruct affected counts only when source fractions and denominators support them; mark reconstructed values as derived.
- Audit each source's `applied_to_treat` and `treats` semantics. A predicate label is not proof of controlled clinical efficacy.

## 3. Family-size weighting

Use reported `n` to retain the information in larger family reports without pretending the data are individual patient records.

- Produce one canonical family feature table with affected fractions, observed masks, feature-specific denominators, genotype, treatment exposure, and source IDs.
- Use the verified number of assessed individuals for each phenotype estimate. Do not assume one study size applies to every feature or combine overlapping reports as new patients.
- Where affected counts and denominators are available, summarize phenotype burden with count/denominator weighting and binomial uncertainty, acknowledging within-family correlation. Where only rounded fractions exist, retain their rounding uncertainty.
- For primary family clustering, give each family a normalized weight proportional to its verified family size. Use weighted centroid updates and a weighted objective; do not duplicate a family's row `n` times.
- Report equal-family and square-root-`n` weighting as sensitivity analyses so a few large pedigrees do not silently determine the conclusion.
- Export both number of families and number of reported individuals in each group. Bootstrap/jackknife families as the sampling units, not synthetic individual rows.
- Weighting does not increase the number of independent families, recover unobserved patient-level feature combinations, or remove ascertainment bias.

## 4. Main deliverable A: cluster every family and build a subtype atlas

### Method

- Use the final full-graph autoencoder embeddings for the primary exploratory analysis, retaining treatment information. Label these **full-graph, treatment-informed family clusters**.
- Cluster the 24 L2-normalized family embeddings in their hidden space with weighted k-means, using a small prespecified `k=2..4` comparison and multiple initializations.
- Assign **every family** to one cluster. Export a 24-row assignment table; no family is dropped as noise. Mark boundary/unstable assignments explicitly rather than treating them as confident subtypes.
- Choose the reported partition using a prespecified combination of cross-seed/resampling stability, separation, and interpretability, documenting all examined `k` values. Do not select the partition solely because it creates an appealing clinical story.
- Run at least five predeclared model seeds, family-level resampling, and leave-one-family-out sensitivity. Align cluster labels before comparing assignments.
- Report a co-assignment matrix, per-family assignment frequency, cluster sizes, and stability. Resampling frequency measures assignment robustness, not a patient's probability of belonging to a biological subtype.
- Compare with raw clinical/genomic feature clustering to show whether the GNN adds structure.
- Train a separate treatment-free checkpoint as the key sensitivity analysis. Remove treatment/management edges and treatment-derived inputs during its training, not only at inference. Compare it with the full-graph partition to identify treatment-driven groups.
- Compare genotype-inclusive and genotype-excluded views and the weighting schemes in Section 3.

### Interpretation without an extra classifier

Do not require LightGBM or another gradient-boosted classifier for the main paper. The subtype labels are already created by clustering; predicting those labels with another model mostly measures how well that model imitates the partition.

Instead, provide:

- A family-by-feature heatmap showing phenotype fractions, genotype, missingness, and reported treatment exposure.
- Per-cluster clinical profiles with `n`-weighted and equal-family summaries, denominators, and uncertainty.
- Representative and boundary families with exact supporting statements.
- Feature-group masking: remove a phenotype/anatomy/genotype input group, recompute the embedding, and measure change in cluster-centroid distance. Describe this as model sensitivity, not a causal disease mechanism.
- Clinician-reviewed cluster names supported by the profiles. Keep neutral cluster IDs when clinical names are not justified.

### Datamap

Produce a labeled 24-family UMAP plot and a CED-neighborhood map of relevant variants, phenotypes, treatments, and supporting entities. UMAP is visualization only; clustering happens in hidden space. Record the neighborhood selection rule, seed, parameters, and category legend.

Reuse the evidence-grounded summary design from `../../DAKPGNN/PLAN.md`: configurable OpenRouter model, `OPENROUTER_API_KEY` from the untracked environment, concurrency 2, cached structured responses, and a USD 1 summary cap after explicit approval. Validate every evidence ID. Never send patient records or credentials. DAKPGNN is a design reference, not a completed implementation to import.

Outputs: `family_assignments.parquet`, `family_profiles.parquet`, `cluster_profiles.parquet`, `cluster_stability.parquet`, co-assignment tensor, datamap coordinates, representative evidence, and publication-quality PNG/PDF figures.

## 5. Main deliverable B: known and novel treatment predictions

### Preserve distinct prediction targets

| Target | Meaning of the predicted association | Required presentation |
| --- | --- | --- |
| `applied_to_treat` | A treatment is used or reported for CED under the source's semantics | Exposure/use-association prediction, not proven benefit |
| Concrete `treats` predicate | A treatment association asserted by that source | Separate source-audited treatment-association prediction, not automatic clinical efficacy |
| Family-conditioned association | A treatment association conditioned on a family's known clinical/genomic context | Personalized research hypothesis, not an individual response probability |

Score the predicates separately; do not combine their probabilities or relabel exposure as `treats`. If concrete `treats` labels lack sufficient support, report that limitation and ranks only rather than fabricate a probability model.

### Global and family-specific scoring

- Rank the declared drug candidate universe for CED overall and for each of the 24 families. Keep procedures and supportive management in separate tables.
- Keep all valid observed treatment edges available in the final descriptive graph. Produce one **known-association table** and a separate **novel-candidate table** that excludes already observed links at the relevant target/context.
- A drug used in another family can still be a new candidate for the target family; distinguish **new to family** from **new to CED**. A drug absent from both treatment predicates can also be considered if its permitted graph features support scoring.
- Explicitly include **off-label repurposing candidates**: drugs approved for other indications, not only treatments already linked to CED. Rank their potential `applied_to_treat` and `treats` associations separately for CED overall and each family using drug-target, pathway, phenotype, and other permitted graph evidence. Lack of an existing CED treatment edge must not exclude a drug from consideration.
- Report approval status, approved indication(s), jurisdiction, source/version/date, existing CED use evidence, and whether the proposal is off-label, investigational, or status unknown. Verify regulatory status from authoritative evidence; absence of a graph approval edge is not proof of off-label status. Keep investigational/unapproved compounds in a separate exploratory tier.
- Distinguish a mechanism-directed CED hypothesis from symptomatic management (for example, a predicted association with a specific CED manifestation). Do not present symptom relief as disease modification. Off-label status describes regulatory context, not evidence of efficacy or safety.
- Current family exposure is represented by `population_context_qualifier` on a drug-CED statement. Implement a family/qualifier-conditioned scorer rather than treating the existing global ComplEx score as family-specific:

```text
score(drug, predicate, CED | family_clinical_genomic_context)
```

- Keep the source representation unchanged. The family context uses known phenotype/genotype/anatomy evidence; exclude the target treatment assertion and derived copies/summaries from that context when evaluating its recovery.
- Record contraindications, conflicts, candidate exclusions, and exact source evidence. Unknown safety is not evidence of safety. Keep a contraindicated drug's evidence visible while excluding it from an unflagged candidate list.
- Explain how family-specific ranks differ from the global ranking; do not claim personalization if they are effectively identical.

### Probabilities

Provide calibrated **treatment-association probabilities** when validation supports them. These estimate the declared graph-link target, not the probability that a patient will benefit.

- Calibrate held-out scores separately for distinct predicate targets when support permits, using a shared low-parameter calibrator rather than 24 family calibrators.
- Group family statements together for family-conditioned calibration/evaluation, including family references occurring only in qualifiers.
- Freeze candidate pools, negative-sampling rules, and any sampling weights. With unobserved links used as negatives, probabilities apply to that declared evaluation population, not automatically to real-world treatment prevalence.
- Report raw score, rank, calibrated association probability, calibration support, uncertainty, and probability definition.
- Evaluate reliability diagrams, Brier score, and log loss. If data cannot support reliable calibration, publish raw scores/ranks and explicitly mark probability as unavailable.
- Do not display `sigmoid(score)` as a validated clinical probability or use labels such as `probability_of_benefit`.

### Per-family report

Each report contains clinical/genomic profile, cluster assignment and robustness, known treatment evidence, top new-to-family and new-to-CED candidates including off-label repurposing, regulatory/evidence tier, intended disease or symptom target, association probabilities when accepted, safety flags, and supporting graph explanations.

Outputs: global/family ranking tables, observed evidence, exclusions, calibration predictions/metrics, and 24 concise family scorecards.

## 6. Main deliverable C: explain predictions and test the GNN's contribution

Keep the explanation workload small and clinically readable:

1. Explain subtype profiles with observed clinical contrasts and feature-group masking.
2. Explain selected top drug predictions with source-backed subgraphs and edge/relation ablation.
3. Add Integrated Gradients for semantic feature-group contributions where the scorer supports it.

Use GNNExplainer only if an adapter to the custom encoder is straightforward and validated. GraphSVX, PGExplainer, broad autoencoder-wide SHAP, and multiclass LightGBM SHAP are optional follow-up work, not publication blockers. Attention weights alone are not explanations.

Every explanation needs a named target, source IDs, baseline/masking definition, and a fidelity check: does removing highly attributed evidence change the score more than removing random matched evidence? Report stability across seeds. Never use a held-out target edge to explain its own prediction.

Compare the GNN against relation-specific popularity, raw-feature/nearest-family similarity, text-only, and no-message-passing baselines. Report exact filtered MRR/Hits with support for CED-family and treatment targets. Use fixed held-out splits and show family-held-out results separately from ordinary statement holdout. Predefine baseline clinical/genomic inputs available for held-out-family inference; target treatment information stays excluded.

Keep evaluated models separate from a later all-edge descriptive refit used for the final atlas and hypothesis tables. Do not attach held-out metrics to a model that already trained on those targets.

## 7. Supplement: future-patient JSON matching

Provide a small, version-pinned GA4GH Phenopacket JSON adapter with one synthetic example. Support HPO present/excluded findings, missingness, TGFB1 variants, measurements/units, anatomy, onset, and provenance.

- Map the record into the same named clinical/genomic feature vocabulary used for family profiles.
- Return nearest families, subtype similarities, shared findings, mismatches, and an abstention when terminology coverage or similarity is inadequate.
- Optionally use a deterministic clinical-text template with the same pinned Sentence Transformers model for both patient and family text. Compare like representations; do not directly compare GNN and text vectors from different spaces.
- Validate key-order invariance, excluded versus missing phenotypes, unknown terms, and out-of-distribution inputs. Set abstention thresholds using held-out data, not a favorable demonstration.
- A family with 40% hearing loss does not specify which other findings occurred in the affected individuals. Do not invent individual profiles from family averages.
- Until external individual-level CED data exist, label outputs as research similarity/membership scores, not calibrated patient subtype or treatment-response probabilities.

This prototype demonstrates a future workflow. It is not necessary to establish diagnostic accuracy for the main exploratory paper.

## 8. Paper package and artifacts

All generated artifacts remain under `.ced/reports/<analysis-id>/` on `wenceslaus`:

```text
manifest.json                  # code/data/model versions, seeds, weighting, splits, checksums
families/                      # 24 profiles, assignments, feature definitions, stability
datamap/                       # coordinates, labels, grounded evidence, PNG/PDF
treatments/                    # known/novel global and family ranks, calibration, scorecards
explanations/                  # evidence subgraphs, attributions, fidelity
evaluation/                    # baselines, held-out results, ablations, uncertainty
patient/                       # optional schema profile and synthetic demonstration
paper/                         # generated figures, tables, supplementary exports
```

| Paper item | Content |
| --- | --- |
| Figure 1 | CED graph, family qualification, GNN, weighting, and analysis/evaluation workflow |
| Figure 2 | All 24 families: datamap, assignments, phenotype/genotype heatmap, and stability |
| Figure 3 | Global/family drug predictions, calibration, and representative evidence subgraphs |
| Table 1 | Family sizes, variants, clinical profiles, completeness, and cluster assignments |
| Table 2 | Candidate subtype profiles and weighting/treatment-free sensitivity results |
| Table 3 | Known and novel treatment associations with accepted probabilities and evidence |
| Supplement | Baseline metrics, all drug ranks, 24 scorecards, methodological details, optional patient JSON example |

Primary claims: exploratory CED family stratification and interpretable treatment-association prediction. Phenotype completion can be a small secondary link-prediction analysis. Treatment response, prognosis, survival, and comparative effectiveness require outcomes not available in these aggregate tables and are outside this paper's current scope.

Follow current JPM author instructions, source licenses, ethics/privacy requirements, and AI-use disclosure policy. Seek CED clinical review of subtype names and drug evidence. Use TRIPOD+AI where applicable to prediction reporting without claiming that a checklist supplies clinical validation.

## 9. Implementation order and acceptance

1. Fix semantic inputs, proportion/zero/missing handling, denominators, and embedding export.
2. Implement family-size-weighted clustering, assign all 24 families, and generate direct clinical profiles.
3. Implement separate predicate and family-conditioned treatment scoring, baselines, and calibration.
4. Freeze evaluation, then build the descriptive atlas, scorecards, and concise explanations.
5. Generate the three main figures/tables and optional patient-matching supplement.

Acceptance requires offline tests for weighting, missing/zero semantics, qualifier-aware family splits, target leakage, deterministic exports, cluster-label alignment, insufficient calibration support, candidate novelty, and explanation fidelity. Run `uv run --no-sync ced check` through the existing remote workflow before full analyses.

All tests, checks, clustering, calibration, plots, and CPU work run on `wenceslaus` with at most four workers and matching thread caps. GPU training/inference runs on Modal. Full builds, preparation, paid GPU jobs, and paid OpenRouter calls require separate explicit approval. No new analysis CLI is claimed to exist yet.

## 10. Research anchors and limitations

Research used Firecrawl and Hindsight plus direct repository inspection. Full-page retrieval of PMC/GeneReviews and MDPI was blocked by browser challenges; verify those full texts and current journal instructions before submission.

- CED family source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC2564495/> and the curated family YAML under `src/ced/graphs/tables/families/`.
- Clinical context: <https://www.ncbi.nlm.nih.gov/books/NBK1156/> and the curated GeneReviews table under `src/ced/graphs/tables/books/`.
- Resampling/consensus clustering: <https://doi.org/10.1023/A:1023949509487>.
- Cluster-wise stability: <https://www.sciencedirect.com/science/article/pii/S0167947306004622>.
- Calibration: <https://proceedings.mlr.press/v70/guo17a.html>. Calibration does not change an association target into a clinical-benefit target.
- Integrated Gradients: <https://proceedings.mlr.press/v70/sundararajan17a.html>.
- GNNExplainer: <https://arxiv.org/abs/1903.03894> and <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.GNNExplainer.html>.
- Phenopacket JSON and observed/excluded semantics: <https://phenopacket-schema.readthedocs.io/en/latest/> and <https://phenopacket-schema.readthedocs.io/en/latest/phenotype.html>. Pin a released schema rather than the mutable documentation version.
- JPM instructions: <https://www.mdpi.com/journal/jpm/instructions>; prediction reporting: <https://www.bmj.com/content/385/bmj-2023-078378>.
- Datamap and environment-key design reference: `../../DAKPGNN/PLAN.md`.
