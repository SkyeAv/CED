---
name: family-table
description: Author or extend curated Tablassert family tables for Camurati-Engelmann disease under src/ced/graphs/tables/families/. Use when the user pastes a clinical table (GeneReviews/PMC screenshots, OCR text, odds ratios) and asks to encode families, phenotypes, mutations, locations, or treatments as table YAML, to plan what to encode, or to update an existing family table.
---

# Family table authoring

Encode user-supplied clinical tables as Tablassert table YAML in
`src/ced/graphs/tables/families/`. One table file per phenotype, location,
mutation overview, or treatment set. Existing files are the schema reference;
there is no other documentation.

## Procedure

1. Read the nearest semantic sibling before writing: phenotype tables
   (`pain-in-extermeties.yaml`, `clinical-manifestation.yaml`), location tables
   (`locations.yaml`, `*-radiological-abnormalities.yaml`), mutation tables
   (`dna-mutations.yaml`, `protein-mutations.yaml`, `mutation-overview.yaml`),
   treatments (`treatments.yaml`). Copy the `template:` block conventions
   verbatim (placeholder TSV source, PMC provenance, annotations).
2. For a large or ambiguous table, first present a plan: which statements will
   be encoded, which skipped, and the subject/object/predicate per row. Wait
   for approval before encoding.
3. Check existing tables for statements already encoded; do not duplicate them.
4. Write the YAML with one `sections:` entry per family or row group, keeping
   study sizes, odds ratios, and denominators exactly as given. Do not invent
   values for missing cells; omit the annotation instead. Pair `effect_type`
   with a numeric `effect_size`: an unpaired `effect_type` is dropped by the
   build (`UnpairedEffectAnnotationWarning`), though the section is kept.
5. Do not run Tablassert validation or compilation for these tables; a custom
   parser is planned and the user explicitly deferred validation. YAML lint via
   `ced check` is the only gate.
6. Do not modify study sizes, odds ratios, or the shared template conventions
   of existing tables without asking; table content is scientific record.

## Conventions in force (from the 49 existing tables)

- Predicates: `has_phenotype` (family has phenotype), `located_in`/`location_of`
  (anatomy), `has_mutation` (mutations), `applied_to_treat` (treatments);
  `clinical-manifestation.yaml` inverts one phenotype set as `affects`
  (subject disease, object family).
- Phenotype rows: subject is the family (`Family 1`), object is the phenotype
  text; prioritize `PhenotypicFeature`. Treatment rows: subject is the
  treatment, prioritize `Treatment`; object is the disease.
- Statement endpoints carry `taxon_from: species_context_qualifier`.
- Qualifiers: `species_context_qualifier` encoding `homo sapiens` prioritizing
  `OrganismTaxon`; `disease_context_qualifier` encoding
  `camurati-englemann disease` prioritizing `Disease`;
  `population_context_qualifier` prioritizing
  `PopulationOfIndividualOrganisms` where the source gives a population.
- Annotations: `study_size`, `effect_size`, `effect_type` encoding `odds_ratio`.
- Provenance: `repo: PMC` plus the `publication:` PMCID of the source article.
- Preserve arbitrary predicates and qualifiers; never force Biolink mappings
  that the source does not state (repo AGENTS.md convention).
