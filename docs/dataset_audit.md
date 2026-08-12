# Dataset Audit

## Dataset Size
- Total images discovered: 7023
- Valid readable images: 7023
- Corrupt/unreadable images: 0
- Zero-byte files: 0

## Current Train/Test Distribution
- Train: 5712
- Test: 1311

## Per-Class Distribution
- glioma: 1621
- meningioma: 1645
- no_tumor: 2000
- pituitary: 1757

## Corruption Findings
- Unreadable/corrupt images: 0
- Non-image extension files: 0

## Exact Duplicate Findings
- Exact duplicate groups: 194
- Cross-split exact duplicate groups: 79
- Cross-class exact duplicate groups: 0

## Near-Duplicate Findings
- Candidate near-duplicate pairs: 18037
- Cross-split near-duplicate pairs: 5557
- Cross-class near-duplicate pairs: 1057
- Algorithm: dhash-8
- Distance threshold: 4

## Subject-Level Leakage
- Reliable subject IDs were not detected from the current legacy filenames.
- Subject-level leakage cannot be ruled out for this dataset.

## Dataset Provenance
- dataset_provenance: UNKNOWN
- Notes: No explicit source URL, license, author, citation, or dataset README establishing provenance was found in the repository files inspected.

## Chosen Split Methodology
- Strategy: rebuild_full_dataset_stratified_80_10_10
- Justification: Cross-split or cross-class exact duplicate leakage was detected, so the legacy split boundaries cannot be trusted safely. The full valid dataset is re-split deterministically using duplicate-aware image-level grouping.

## Final Train/Val/Test Counts
- Train: 5618
- Val: 703
- Test: 702

## Final Per-Class Split Counts
- train: {'glioma': 1297, 'meningioma': 1317, 'no_tumor': 1599, 'pituitary': 1405}
- val: {'glioma': 162, 'meningioma': 164, 'no_tumor': 201, 'pituitary': 176}
- test: {'glioma': 162, 'meningioma': 164, 'no_tumor': 200, 'pituitary': 176}

## Known Limitations
- Subject-level leakage cannot be excluded because reliable subject identifiers were not available.
- Near-duplicate detection uses perceptual hashing and should be treated as a review aid, not ground truth.
- Provenance, license, and citation information were not established from the current repository contents.

## Dataset Fingerprint
- dataset_fingerprint: `b15363ff85ecf29d11c67a2c38fad723a7bb339cb4489e53dd637ab16f8c24b1`
