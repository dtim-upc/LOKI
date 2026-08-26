# Selection Methodology & Statistical Analysis

## 1. Goal
Select 111 patients (≈ 400 admissions) from the `mimic_split/test` dataset to create a balanced corpus for annotating four clinical relationship types:
- `ADVERSE_EFFECT`
- `CONTRAINDICATED`
- `DISCONTINUED`
- `TREATS`

The selection procedure is intentionally biased toward the three rarer, clinically safer-critical classes (`ADVERSE_EFFECT`, `CONTRAINDICATED`, `DISCONTINUED`). `TREATS` is structurally dominant in any clinical discharge note and does not require targeted candidate selection — the LLM annotator will surface these naturally regardless of which patient is chosen.

---

## 2. Scoring Strategy

Each patient admission (`hadm_id`) receives a score per class, computed by `1_generate_stats.py`. The scoring strategy differs per class based on the failure modes discovered during empirical analysis of the test set.

### 2.1 Failure Mode Analysis (v1 Approach)

The original approach scanned raw note text for keywords: `\bfor\b` for TREATS, `contraindicated`/`avoid` for CONTRAINDICATED, etc. This produced severely skewed results:

| Class | Problem |
| :--- | :--- |
| **TREATS** | `\bfor\b` fired 10–25 times per note on non-therapeutic sentences ("admitted for...", "scheduled for...", "labs ordered for..."), causing TREATS to dominate selection |
| **CONTRAINDICATED** | "avoid" and "contraindicated" appear in generic patient-education text ("avoid strenuous activity") with no link to a specific drug or diagnosis from the patient's own tables |
| **ADVERSE_EFFECT** | Missing the `-induced` pattern ("steroid-induced hyperglycemia"), which is the most lexically distinctive MIMIC signal for this class |
| **DISCONTINUED** | Keywords fired on irrelevant text ("held for clinic appointment") with no nearby drug name |

Reflecting analysis on the 18-admission small annotated set confirmed: 0 CONTRAINDICATED relationships were found despite all selected patients scoring on the "contraindicated/avoid" keywords.

### 2.2 Current Approach: Class-Specific Strategies

Each class now uses a fundamentally different scoring mechanism matched to how that relationship type manifests in clinical text.

#### ADVERSE_EFFECT — Split entity-anchored keyword scoring

Patterns are divided into two groups:

**Anchored patterns** (only counted when a drug name from this patient's `medication.csv` also appears in the same sentence):
- `secondary to`, `due to`, `caused by`, `after starting`, `attributed to`, `as a result of`

**Self-anchored patterns** (counted in any sentence — the drug is embedded in the pattern itself):
- `\w+-induced` (e.g. "steroid-induced hyperglycemia", "chemotherapy-induced nausea")
- `drug-induced`, `medication-induced`

This prevents false positives like "AKI due to dehydration" (no drug present) while catching the common MIMIC pattern "steroid-induced hyperglycemia" even when the specific steroid name appears elsewhere in the sentence.

#### DISCONTINUED — Entity-anchored keyword scoring

Only counted when both a keyword AND a drug name from this patient's `medication.csv` appear in the same sentence. Patterns include status-change verbs with and without their causal connectors to capture both the bare fact and the reason:

- `held due to`, `held because`, `held in the setting of`
- `stopped due to`, `stopped because`
- `\bheld\b`, `\bdiscontinued\b`, `\bstopped\b`
- `converted to`, `switched to`, `transitioned to`

Entity-anchoring filters out "held for clinic appointment", "stopped smoking", "transitioned to rehab", etc.

#### TREATS — Deliberately suppressed via specific phrases only (entity-anchored)

`\bfor\b` is intentionally excluded. Only high-specificity therapeutic-intent phrases are used:
- `started for`, `prescribed for`, `indicated for`
- `to treat`, `to manage`, `therapy for`, `treatment for`

These phrases are rare in MIMIC notes, so TREATS scores stay very low by design. The purpose is that the TREATS class fills only the 4th selection pass (residual patients), preventing TREATS-rich notes from displacing patients with rarer relationships. TREATS relationships will appear abundantly during actual LLM annotation regardless — they do not need to be emphasized at the selection stage.

#### CONTRAINDICATED — Knowledge-base table lookup (no text scanning)

Text scanning is abandoned entirely for this class. The core insight is: if a drug is truly contraindicated, it often does not appear in the discharge medication list at all — meaning "contraindicated" in the note rarely refers to a drug that is both (a) present in the patient's own medication table and (b) co-occurring with the contraindicated diagnosis.

Instead, each admission is scored by cross-referencing the patient's actual `medication.csv` against their `diagnosis.csv` using a curated knowledge base of 36 clinically established drug–disease contraindication pairs. A pair scores if:
- Any drug in the medication table contains the drug substring, **AND**
- Any diagnosis in the diagnosis table contains at least one condition substring

Each KB entry counts at most once per admission. Score = number of confirmed pairs.

**Knowledge Base Coverage (selected examples)**:

| Drug | Contraindicated Conditions |
| :--- | :--- |
| Metformin | Acute kidney injury, CKD, renal failure |
| Metoprolol / Carvedilol / Atenolol | Asthma, reactive airway, bronchospasm, decompensated heart failure |
| Warfarin | Intracranial hemorrhage, subdural hematoma, active bleed |
| Digoxin | Heart block, AV block, bradycardia, sinus pause |
| Spironolactone / Eplerenone | Hyperkalemia, renal failure |
| Lisinopril / Enalapril / Ramipril | Angioedema, hyperkalemia, renal artery stenosis |
| Celecoxib / Ibuprofen / Naproxen | GI bleed, renal failure, heart failure |
| Pioglitazone / Rosiglitazone | Heart failure, CHF |
| Vancomycin / Gentamicin | Acute kidney injury, CKD |
| Tramadol | Seizure, epilepsy |
| Ciprofloxacin / Levofloxacin | QT prolongation, myasthenia gravis |
| Potassium / Trimethoprim | Hyperkalemia |
| Heparin | Heparin-induced thrombocytopenia (HIT) |
| *(+ 23 more pairs)* | |

This approach selects patients where a known contraindication pair is structurally present in their structured data, making it highly probable that the LLM annotator will find and label a genuine `CONTRAINDICATED` relationship.

---

## 3. Statistical Analysis of Test Set

### 3.1 Method
`1_generate_stats.py` scanned all **1,242 note files** across **659 patients** in the `mimic_split/test` set. Scores are computed at the admission (`hadm_id`) level and then aggregated per patient (summed across all visits).

### 3.2 Findings (Patient-Level Distribution)

| Relation Type | Scoring Method | Patients with Score > 0 | Availability |
| :--- | :--- | :--- | :--- |
| **ADVERSE_EFFECT** | Entity-anchored keywords + self-anchored `-induced` | **101** (15.3%) | Sufficient |
| **CONTRAINDICATED** | Knowledge-base table lookup | **155** (23.4%) | Moderate |
| **DISCONTINUED** | Entity-anchored keywords | **231** (34.9%) | Sufficient |
| **TREATS** | Specific phrases only (suppressed) | **38** (5.7%) | Intentionally rare |

**Key insight**: All three priority classes have sufficient candidates (≥101 patients) to fill 25 slots with patients that have meaningfully high scores. The TREATS pool is small by design — the 4th pass simply selects whoever remains.

---

## 4. Selection Method

`2_extract_patients.py` implements **Stratified Priority Selection**: rarest/hardest classes are filled first using top-scoring patients, with mutual exclusion enforced across passes.

### 4.1 Algorithm

The script operates in **`score_target` mode**: instead of a fixed patient count per class, it selects the *minimum* number of patients from each priority class needed to reach a shared score target, then fills the remainder with TREATS-filler patients.

**Auto target computation**: A simulation pass selects the top `N_PER_CLASS=25` patients per class (with mutual exclusion) and records the cumulative score each class achieves. The shared target is set to the minimum across the three classes — in the current run, `min(AE=99, CI=111, DISC=129) = 99`.

1. **Pass 1 (ADVERSE_EFFECT)**: Sort all 659 patients by `ADVERSE_EFFECT` score (desc). Add patients until cumulative score ≥ 99. → **25 patients, score = 99**.
2. **Pass 2 (CONTRAINDICATED)**: Remove Pass 1 patients. Sort remaining by `CONTRAINDICATED` score. Add until score ≥ 99. → **21 patients, score = 99**.
3. **Pass 3 (DISCONTINUED)**: Remove prior selections. Sort by `DISCONTINUED` score. Add until score ≥ 99. → **15 patients, score = 101**.
4. **Pass 4 (TREATS)**: Fill remaining budget (`TARGET_COUNT − 61 = 39`) from whoever is left, sorted by `TREATS` score. → **39 patients**.

### 4.2 Score Ranges of Selected Patients

| Class | Patients | Total Score | Min | Max | Mean | Total Admissions | Evidence Density |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ADVERSE_EFFECT** | 25 | 99 | 2 | 17 | 3.96 | 125 | 22.6% |
| **CONTRAINDICATED** | 21 | 99 | 3 | 11 | 4.71 | 81 | 29.5% |
| **DISCONTINUED** | 15 | 101 | 4 | 12 | 6.73 | 78 | 40.8% |
| **TREATS** | 39 | 30 | 0 | 2 | 0.77 | 96 | 10.5% |
| **Total** | **100** | — | — | — | — | **380** | — |

All three priority classes converge at cumulative score ≈ 99–101, meaning approximately equal expected relationship yield per class. The score_target mode achieves this automatically without a fixed patient cap — DISCONTINUED requires only 15 patients to match AE's yield because DISCONTINUED patients score ~1.7× higher per patient (mean 6.73 vs 3.96), reflecting multi-admission high-signal cases.

### 4.3 Reasoning

- **Why stratified?** Random selection would yield ~95% `TREATS`-dominant patients under the suppressed scoring scheme, and even under any scheme TREATS is the structurally most common class. Stratification guarantees representation of the three safety-critical classes.
- **Why score_target instead of fixed N?** Fixed N over-selects high-per-patient-score classes. A DISCONTINUED patient averages 6.73 signal per patient vs. 4.0 for ADVERSE_EFFECT — selecting 25 DISCONTINUED patients would produce 129 expected relationships vs. 100 for AE. Score_target equalises expected yield, which means fewer DISCONTINUED slots but proportionally more TREATS-filler diversity.
- **Why priority sort?** Top-scoring patients maximize the probability that the LLM annotator will actually find valid instances of that relationship type, not just that the keywords/pairs were present.
- **Why mutually exclusive passes?** Ensures 111 unique patients, maximizing diversity of clinical contexts.
- **Why suppress TREATS in scoring?** The LLM annotator will label TREATS relationships abundantly in every note regardless of which patient is chosen. Suppressing TREATS in the scoring prevents TREATS-rich patients from occupying slots that rare-class patients should fill.

---

## 5. Output

The final result is `outputs/selected_patients.csv` containing **100 unique patient IDs across 380 admissions**:

- **25** high-probability `ADVERSE_EFFECT` candidates (entity-anchored causal language near patient's own drugs)
- **21** high-probability `CONTRAINDICATED` candidates (≥3 known drug–disease contraindication pairs in their structured tables)
- **15** high-probability `DISCONTINUED` candidates (entity-anchored status-change language near patient's own drugs)
- **39** filler `TREATS` candidates (residual patients — will yield TREATS annotations naturally during LLM annotation)

All three priority classes contribute approximately 99–101 expected relationship instances to the final annotated corpus, ensuring balanced training signal despite the structural rarity of each class.
