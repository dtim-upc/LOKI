"""
sample_test_for_annotation.py

Single-script replacement for Class_Balancing/1_generate_stats.py +
Class_Balancing/2_extract_patients.py.

Pipeline:
  1. Scan all admissions in TEST_DIR and score each (subject_id, hadm_id)
     on four relationship classes (ADVERSE_EFFECT, CONTRAINDICATED,
     DISCONTINUED, TREATS).
  2. Aggregate scores per patient, then select TARGET_COUNT patients using
     Stratified Priority Selection so that the three rare classes reach an
     equal expected relationship yield.

Outputs (all written to OUTPUT_DIR):
  patient_relation_scores.csv  — admission-level scores (1 row per hadm_id)
  selected_patients.csv        — selected patient IDs with selection reason
  selection_stats.json         — detailed statistics of the final selection
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


# =============================================================================
# ENTITY-ANCHORED KEYWORD PATTERNS
#
# Strategy: keywords for TREATS, ADVERSE_EFFECT, and DISCONTINUED are matched
# at the sentence level, and only counted when the sentence also contains a
# drug name from the patient's own medication table (entity-anchoring).
#
# This eliminates high-frequency false positives such as:
#   - "admitted for further management"  (TREATS: \bfor\b)
#   - "avoid strenuous activity"          (CONTRAINDICATED: avoid)
#   - "held for clinic appointment"       (DISCONTINUED: held)
#
# CONTRAINDICATED uses an entirely separate knowledge-base approach (see below).
# =============================================================================

# ADVERSE_EFFECT patterns are split into two groups:
#   ANCHORED      — require a drug token in the same sentence
#   SELF_ANCHORED — the drug reference is inside the pattern itself
#     (e.g. "steroid-induced", "drug-induced") — counted without extra drug token
AE_ANCHORED_PATTERNS = [
    r"secondary to",
    r"due to",
    r"caused by",
    r"after starting",
    r"attributed to",
    r"as a result of",
]
AE_SELF_ANCHORED_PATTERNS = [
    r"\w+-induced",   # "steroid-induced", "chemotherapy-induced", etc.
    r"drug-induced",
    r"medication-induced",
]

KEYWORDS = {
    "DISCONTINUED": [
        r"held due to",
        r"held because",
        r"held in the setting of",
        r"stopped due to",
        r"stopped because",
        r"\bheld\b",
        r"\bdiscontinued\b",
        r"\bstopped\b",
        r"converted to",
        r"switched to",
        r"transitioned to",
    ],
    "TREATS": [
        r"started for",
        r"prescribed for",
        r"indicated for",
        r"to treat",
        r"to manage",
        r"therapy for",
        r"treatment for",
    ],
}

# =============================================================================
# CONTRAINDICATED — KNOWLEDGE-BASE TABLE LOOKUP
#
# For each admission we cross-reference the patient's actual medication table
# against their diagnosis table.  A pair scores if:
#   (a) any drug in the medication table contains the drug_substring, AND
#   (b) any diagnosis in the diagnosis table contains at least one of the
#       condition_substrings.
#
# Score = number of such confirmed drug-diagnosis contraindication pairs.
# Note text is NOT scanned for CONTRAINDICATED at all.
# =============================================================================

# Each entry: (drug_substring, [condition_substrings])
CONTRAINDICATION_KB = [
    ("metformin",           ["renal failure", "acute kidney", "ckd", "renal insufficiency",
                              "renal impairment", "kidney disease", "chronic kidney"]),
    ("ibuprofen",           ["renal failure", "acute kidney", "gi bleed", "gastrointestinal hemorrhage",
                              "gastrointestinal bleed", "heart failure", "peptic ulcer"]),
    ("celecoxib",           ["renal failure", "acute kidney", "gi bleed", "gastrointestinal hemorrhage",
                              "gastrointestinal bleed", "heart failure"]),
    ("naproxen",            ["renal failure", "acute kidney", "gi bleed", "gastrointestinal hemorrhage",
                              "heart failure"]),
    ("indomethacin",        ["renal failure", "acute kidney", "gi bleed", "heart failure"]),
    ("warfarin",            ["intracranial hemorrhage", "subdural hematoma", "epidural hematoma",
                              "active bleed", "hemorrhage", "gastrointestinal hemorrhage"]),
    ("heparin",             ["heparin-induced thrombocytopenia", "hit"]),
    ("digoxin",             ["heart block", "av block", "atrioventricular block", "bradycardia",
                              "sinus pause", "sick sinus"]),
    ("metoprolol",          ["asthma", "reactive airway", "bronchospasm",
                              "decompensated heart failure", "acute decompensated"]),
    ("carvedilol",          ["asthma", "reactive airway", "bronchospasm",
                              "decompensated heart failure", "acute decompensated"]),
    ("atenolol",            ["asthma", "reactive airway", "bronchospasm"]),
    ("propranolol",         ["asthma", "reactive airway", "bronchospasm"]),
    ("lisinopril",          ["angioedema", "hyperkalemia", "renal artery stenosis",
                              "bilateral renal artery"]),
    ("enalapril",           ["angioedema", "hyperkalemia", "renal artery stenosis"]),
    ("ramipril",            ["angioedema", "hyperkalemia", "renal artery stenosis"]),
    ("captopril",           ["angioedema", "hyperkalemia"]),
    ("spironolactone",      ["hyperkalemia", "renal failure", "acute kidney"]),
    ("eplerenone",          ["hyperkalemia", "renal failure", "acute kidney"]),
    ("pioglitazone",        ["heart failure", "congestive heart failure", "chf",
                              "bladder cancer"]),
    ("rosiglitazone",       ["heart failure", "congestive heart failure", "chf"]),
    ("amiodarone",          ["thyroid disorder", "hypothyroidism", "hyperthyroidism",
                              "thyrotoxicosis", "pulmonary toxicity", "interstitial lung"]),
    ("gentamicin",          ["renal failure", "acute kidney", "ckd"]),
    ("tobramycin",          ["renal failure", "acute kidney", "ckd"]),
    ("vancomycin",          ["renal failure", "acute kidney", "ckd"]),
    ("clopidogrel",         ["active bleed", "hemorrhage", "intracranial hemorrhage"]),
    ("aspirin",             ["active bleed", "gi bleed", "gastrointestinal hemorrhage",
                              "peptic ulcer", "gastrointestinal bleed"]),
    ("tramadol",            ["seizure", "epilepsy", "seizure disorder"]),
    ("ciprofloxacin",       ["qt prolongation", "long qt", "myasthenia gravis"]),
    ("levofloxacin",        ["qt prolongation", "long qt", "myasthenia gravis"]),
    ("moxifloxacin",        ["qt prolongation", "long qt"]),
    ("potassium",           ["hyperkalemia"]),
    ("trimethoprim",        ["hyperkalemia", "renal failure", "acute kidney"]),
    ("lithium",             ["renal failure", "acute kidney", "ckd"]),
    ("colchicine",          ["renal failure", "acute kidney", "ckd"]),
    ("allopurinol",         ["renal failure", "acute kidney", "ckd"]),
    ("nitrofurantoin",      ["renal failure", "acute kidney", "ckd", "chronic kidney"]),
]

# =============================================================================
# HELPERS
# =============================================================================

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|[\n]+')

def split_sentences(text):
    raw = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in raw if len(s.strip()) > 5]


def load_drug_tokens(hadm_folder, hadm_id):
    med_path = os.path.join(hadm_folder, f"{hadm_id}-medication.csv")
    tokens = set()
    if not os.path.exists(med_path):
        return tokens
    try:
        with open(med_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_name = (row.get("drug") or "").strip().lower()
                if not drug_name:
                    continue
                tokens.add(drug_name)
                first_word = drug_name.split()[0]
                if len(first_word) >= 4:
                    tokens.add(first_word)
    except Exception:
        pass
    return tokens


def load_diagnoses(hadm_folder, hadm_id):
    diag_path = os.path.join(hadm_folder, f"{hadm_id}-diagnosis.csv")
    diagnoses = []
    if not os.path.exists(diag_path):
        return diagnoses
    try:
        with open(diag_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                diag = (row.get("diagnosis") or "").strip().lower()
                if diag:
                    diagnoses.append(diag)
    except Exception:
        pass
    return diagnoses


def score_anchored_keywords(note_text, drug_tokens):
    counts = {"ADVERSE_EFFECT": 0, "DISCONTINUED": 0, "TREATS": 0}
    for sentence in split_sentences(note_text):
        sent_lower = sentence.lower()

        # Self-anchored AE patterns fire in any sentence
        for pattern in AE_SELF_ANCHORED_PATTERNS:
            if re.search(pattern, sent_lower):
                counts["ADVERSE_EFFECT"] += 1
                break

        # Drug-anchored scoring requires a drug token present in this sentence
        if not drug_tokens:
            continue
        if not any(tok in sent_lower for tok in drug_tokens):
            continue

        # Anchored AE patterns
        for pattern in AE_ANCHORED_PATTERNS:
            if re.search(pattern, sent_lower):
                counts["ADVERSE_EFFECT"] += 1
                break

        # DISCONTINUED and TREATS — one hit per (sentence, class)
        for relation in ("DISCONTINUED", "TREATS"):
            for pattern in KEYWORDS[relation]:
                if re.search(pattern, sent_lower):
                    counts[relation] += 1
                    break

    return counts


def score_contraindications(drug_tokens, diagnoses):
    score = 0
    for drug_sub, condition_subs in CONTRAINDICATION_KB:
        if not any(drug_sub in tok for tok in drug_tokens):
            continue
        if any(cond_sub in diag for diag in diagnoses for cond_sub in condition_subs):
            score += 1
    return score


# =============================================================================
# STAGE 1 — Generate per-admission scores
# =============================================================================

def generate_scores():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("STAGE 1: Scoring admissions")
    print(f"{'='*60}")
    print(f"Scanning {TEST_DIR} for patient notes...")

    note_files = glob.glob(os.path.join(TEST_DIR, "**", "*-notes.txt"), recursive=True)
    print(f"Found {len(note_files)} note files.")

    results = []
    for i, file_path in enumerate(note_files):
        try:
            norm_path = file_path.replace("\\", "/")
            parts = norm_path.split("/")
            hadm_id = parts[-2]
            subject_id = parts[-3]
            hadm_folder = os.path.dirname(file_path)

            with open(file_path, "r", encoding="utf-8-sig") as f:
                note_text = f.read()

            drug_tokens = load_drug_tokens(hadm_folder, hadm_id)
            anchored = score_anchored_keywords(note_text, drug_tokens)
            diagnoses = load_diagnoses(hadm_folder, hadm_id)
            contraindicated_score = score_contraindications(drug_tokens, diagnoses)

            results.append({
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "ADVERSE_EFFECT": anchored["ADVERSE_EFFECT"],
                "CONTRAINDICATED": contraindicated_score,
                "DISCONTINUED": anchored["DISCONTINUED"],
                "TREATS": anchored["TREATS"],
            })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(note_files)} files...")

        except Exception as e:
            print(f"  Error reading {file_path}: {e}")

    fieldnames = ["subject_id", "hadm_id", "ADVERSE_EFFECT", "CONTRAINDICATED",
                  "DISCONTINUED", "TREATS"]
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {SCORES_CSV}")

    print("\n--- Score Distribution (admission-level) ---")
    for col in ["ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED", "TREATS"]:
        values = [r[col] for r in results]
        nonzero = sum(1 for v in values if v > 0)
        print(f"  {col:20s}: mean={sum(values)/len(values):.2f}  "
              f"admissions_with_score>0={nonzero}/{len(results)}")

    return results


# =============================================================================
# STAGE 2 — Select patients
# =============================================================================

def select_patients(visits):
    print(f"\n{'='*60}")
    print("STAGE 2: Selecting patients")
    print(f"{'='*60}")

    # Aggregate by patient
    patient_scores = defaultdict(lambda: {
        "ADVERSE_EFFECT": 0, "CONTRAINDICATED": 0, "DISCONTINUED": 0, "TREATS": 0,
        "visits": []
    })
    for v in visits:
        sid = v["subject_id"]
        patient_scores[sid]["visits"].append(v["hadm_id"])
        for k in ["ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED", "TREATS"]:
            patient_scores[sid][k] += int(v[k])

    print(f"Aggregated into {len(patient_scores)} unique patients.")
    patients = [{"subject_id": k, **v} for k, v in patient_scores.items()]

    selected = []
    selected_ids = set()

    def add_candidate(candidate, reason):
        if candidate["subject_id"] not in selected_ids:
            selected.append({
                "subject_id": candidate["subject_id"],
                "selection_reason": reason,
                "reason_score": candidate[reason],
                "total_visits": len(candidate["visits"])
            })
            selected_ids.add(candidate["subject_id"])
            return True
        return False

    def run_pass(class_key, all_patients, n_limit, score_limit=None):
        pool = sorted(
            [p for p in all_patients if p["subject_id"] not in selected_ids and p[class_key] > 0],
            key=lambda x: x[class_key], reverse=True
        )
        count = 0
        cum_score = 0
        for c in pool:
            if count >= n_limit:
                break
            if score_limit is not None and cum_score >= score_limit:
                break
            if add_candidate(c, class_key):
                cum_score += c[class_key]
                count += 1
        return count, cum_score

    # ── Determine effective target (score_target mode) ─────────────────────
    effective_target = None
    if BALANCE_MODE == "score_target":
        if SCORE_TARGET is not None:
            effective_target = SCORE_TARGET
            print(f"Score-target mode: using user-specified target = {effective_target}")
        else:
            sim_selected = set()
            sim_scores = {}
            for cls in ["ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED"]:
                pool = sorted(
                    [p for p in patients if p["subject_id"] not in sim_selected and p[cls] > 0],
                    key=lambda x: x[cls], reverse=True
                )
                cum = sum(p[cls] for p in pool[:N_PER_CLASS])
                for p in pool[:N_PER_CLASS]:
                    sim_selected.add(p["subject_id"])
                sim_scores[cls] = cum
            effective_target = min(sim_scores.values())
            print(f"Score-target mode (auto): reference scores at N={N_PER_CLASS}: "
                  f"AE={sim_scores['ADVERSE_EFFECT']}  "
                  f"CI={sim_scores['CONTRAINDICATED']}  "
                  f"DISC={sim_scores['DISCONTINUED']}")
            print(f"  → target set to minimum = {effective_target}")

    # ── Priority passes ────────────────────────────────────────────────────
    priority_classes = ["ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED"]
    n_selected_per_class = {}
    score_per_class = {}

    for cls in priority_classes:
        if BALANCE_MODE == "fixed":
            n, score = run_pass(cls, patients, n_limit=N_PER_CLASS)
        else:
            n, score = run_pass(cls, patients, n_limit=TARGET_COUNT, score_limit=effective_target)
        n_selected_per_class[cls] = n
        score_per_class[cls] = score
        print(f"Selected {n} patients for {cls}  (cumulative score={score}).")

    # ── TREATS filler ──────────────────────────────────────────────────────
    _score_keys = ["ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED", "TREATS"]
    pool_treats = sorted(
        [p for p in patients if p["subject_id"] not in selected_ids and p["TREATS"] > 0],
        key=lambda x: x["TREATS"], reverse=True
    )
    # Fallback: unselected patients with at least one non-zero score (any class), sorted by TREATS
    pool_remainder = pool_treats + sorted(
        [p for p in patients if p["subject_id"] not in selected_ids
         and p not in pool_treats
         and any(p[k] > 0 for k in _score_keys)],
        key=lambda x: x["TREATS"], reverse=True
    )
    n_treats = 0
    for c in pool_remainder:
        if len(selected) >= TARGET_COUNT:
            break
        if add_candidate(c, "TREATS"):
            n_treats += 1
    n_selected_per_class["TREATS"] = n_treats
    score_per_class["TREATS"] = sum(
        s["reason_score"] for s in selected if s["selection_reason"] == "TREATS"
    )
    print(f"Selected {n_treats} patients for TREATS  (cumulative score={score_per_class['TREATS']}).")
    print(f"Total selected: {len(selected)}")

    # ── Write CSV ──────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "selection_reason",
                                                "reason_score", "total_visits"])
        writer.writeheader()
        writer.writerows(selected)
    print(f"Wrote selected patients to {OUTPUT_CSV}")

    # ── Build stats JSON ───────────────────────────────────────────────────
    selected_visits = [v for v in visits if v["subject_id"] in selected_ids]
    CLASSES = ["ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED", "TREATS"]

    def _class_stats(reason_label):
        subset = [s for s in selected if s["selection_reason"] == reason_label]
        scores = [s["reason_score"] for s in subset]
        adm_counts = [s["total_visits"] for s in subset]
        return {
            "patients_selected": len(subset),
            "score_min": min(scores) if scores else 0,
            "score_max": max(scores) if scores else 0,
            "score_mean": round(sum(scores) / len(scores), 2) if scores else 0,
            "score_total": sum(scores),
            "admissions_min": min(adm_counts) if adm_counts else 0,
            "admissions_max": max(adm_counts) if adm_counts else 0,
            "admissions_mean": round(sum(adm_counts) / len(adm_counts), 2) if adm_counts else 0,
            "admissions_total": sum(adm_counts),
        }

    agg_selected = defaultdict(lambda: {k: 0 for k in CLASSES})
    for v in selected_visits:
        for k in CLASSES:
            agg_selected[v["subject_id"]][k] += int(v[k])

    total_admissions = sum(s["total_visits"] for s in selected)

    class_prevalence = {}
    for k in CLASSES:
        adm_with_evidence = sum(1 for v in selected_visits if int(v[k]) > 0)
        class_prevalence[k] = {
            "admissions_with_evidence": adm_with_evidence,
            "admissions_with_evidence_pct": round(adm_with_evidence / len(selected_visits) * 100, 1)
                                            if selected_visits else 0,
            "patients_with_evidence": sum(1 for p in agg_selected.values() if p[k] > 0),
        }

    balance_config = {"mode": BALANCE_MODE, "n_per_class_reference": N_PER_CLASS}
    if BALANCE_MODE == "score_target":
        balance_config["score_target"] = effective_target
        balance_config["score_target_source"] = ("user" if SCORE_TARGET is not None
                                                  else "auto (min at N_PER_CLASS)")

    stats = {
        "balance_config": balance_config,
        "overview": {
            "total_patients_selected": len(selected),
            "total_admissions": total_admissions,
            "total_note_files": total_admissions,
            "source_pool_patients": len(patient_scores),
            "source_pool_admissions": len(visits),
        },
        "by_selection_class": {cls: _class_stats(cls) for cls in CLASSES},
        "class_prevalence_in_selected": class_prevalence,
        "per_patient": [
            {
                "subject_id": s["subject_id"],
                "selection_reason": s["selection_reason"],
                "reason_score": s["reason_score"],
                "total_admissions": s["total_visits"],
                "scores": {k: agg_selected[s["subject_id"]][k] for k in CLASSES},
            }
            for s in selected
        ],
    }

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote statistics to {OUTPUT_STATS}")

    # ── Human-readable summary ─────────────────────────────────────────────
    print("\n--- Selection Summary ---")
    ov = stats["overview"]
    print(f"  Mode              : {BALANCE_MODE}" +
          (f" (target score = {effective_target})" if BALANCE_MODE == "score_target" else ""))
    print(f"  Patients selected : {ov['total_patients_selected']} / {ov['source_pool_patients']}")
    print(f"  Total admissions  : {ov['total_admissions']} "
          f"(from {ov['source_pool_admissions']} pool admissions)")
    print(f"  Total note files  : {ov['total_note_files']}")
    print()
    for cls in CLASSES:
        cs = stats["by_selection_class"][cls]
        cp = stats["class_prevalence_in_selected"][cls]
        print(f"  {cls}:")
        print(f"    Selected patients : {cs['patients_selected']}")
        print(f"    Reason score      : min={cs['score_min']}  max={cs['score_max']}  "
              f"mean={cs['score_mean']}  total={cs['score_total']}")
        print(f"    Admissions        : total={cs['admissions_total']}  "
              f"mean={cs['admissions_mean']}")
        print(f"    Evidence density  : {cp['admissions_with_evidence']} admissions "
              f"with score>0 ({cp['admissions_with_evidence_pct']}%)")
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Score MIMIC-IV test admissions and select patients for annotation."
    )
    p.add_argument(
        "--test_dir", default=r"mimic_split/test",
        help="Test split directory to scan (default: mimic_split/test)",
    )
    p.add_argument(
        "--output_dir", default=r"annotation_candidates",
        help="Output directory for all generated files (default: annotation_candidates)",
    )
    p.add_argument(
        "--target_count", type=int, default=100,
        help="Total patients to select (default: 100)",
    )
    p.add_argument(
        "--balance_mode", choices=["fixed", "score_target"], default="score_target",
        help="Patient selection strategy (default: score_target)",
    )
    p.add_argument(
        "--n_per_class", type=int, default=25,
        help="Patients per class in fixed mode / reference N for auto target (default: 25)",
    )
    p.add_argument(
        "--score_target", type=int, default=None,
        help="Explicit score target for score_target mode; omit for auto (default: auto)",
    )
    args = p.parse_args()

    global TEST_DIR, OUTPUT_DIR, SCORES_CSV, OUTPUT_CSV, OUTPUT_STATS
    global TARGET_COUNT, BALANCE_MODE, N_PER_CLASS, SCORE_TARGET
    TEST_DIR     = args.test_dir
    OUTPUT_DIR   = args.output_dir
    SCORES_CSV   = os.path.join(OUTPUT_DIR, "patient_relation_scores.csv")
    OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "selected_patients.csv")
    OUTPUT_STATS = os.path.join(OUTPUT_DIR, "selection_stats.json")
    TARGET_COUNT = args.target_count
    BALANCE_MODE = args.balance_mode
    N_PER_CLASS  = args.n_per_class
    SCORE_TARGET = args.score_target

    visits = generate_scores()
    select_patients(visits)
    print("Done.")


if __name__ == "__main__":
    main()
