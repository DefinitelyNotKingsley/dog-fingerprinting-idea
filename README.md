# Dog Bark Fingerprinting using Metric Learning

## Overview

This project explores **open-set dog bark fingerprinting**, where the goal is to determine whether a bark belongs to a known dog and, if so, identify which dog produced it.

Unlike traditional classification systems that can only recognize dogs seen during training, this project focuses on **fingerprinting**, where a model learns a discriminative embedding space that can:

1. Identify enrolled dogs using only a few bark samples.
2. Reject previously unseen dogs as unknown.
3. Scale to new dogs without retraining the entire model.

The overall design is inspired by modern human speaker recognition and open-set speaker identification systems such as VoxBlink, ECAPA-TDNN, ArcFace, and Triplet Loss based fingerprinting approaches.

---

## Research Motivation

Dogs communicate through vocalizations that contain information about identity, emotional state, and behavior. While dog bark classification has been explored in prior work, relatively little research has investigated whether individual dogs possess unique vocal fingerprints analogous to human voiceprints.

The primary research question is:

> Can individual dogs be identified from bark recordings using learned biometric-style embeddings and open-set identification techniques?

This project investigates that question using metric learning and large-scale evaluation on the DogSpeak dataset.

---

## Dataset

### DogSpeak Dataset

The experiments use the DogSpeak dataset:

* 77,201 bark recordings
* 155 individual dogs
* Multiple breeds
* Multiple recording conditions

Dataset statistics:

| Statistic           | Value                   |
| ------------------- | ----------------------- |
| Total Dogs          | 155                     |
| Total Bark Clips    | 77,201                  |
| Sample Rate         | 16 kHz                  |
| Evaluation Protocol | Open-set identification |

---

## Final Pipeline

The final pipeline consists of three complementary feature extractors followed by metric learning.

```text
Dog Bark Audio
        │
        ├── Dog2Vec (768-D)
        │
        ├── MFCC (80-D)
        │
        └── ECAPA Embedding (192-D)
                │
                ▼
      Feature Fusion Layer
                │
                ▼
         256-D Embedding
                │
                ▼
          Triplet Loss
                │
                ▼
     Dog Fingerprint Space
                │
                ▼
 Enrollment Profiles (1,3,5 barks)
                │
                ▼
       Cosine Similarity
                │
                ▼
      Identify / Reject
```

---

## Feature Extraction

### Dog2Vec

Dog2Vec is a pretrained canine bioacoustic representation model.

Output:

* 768-dimensional embedding

Purpose:

* Captures species-specific bark characteristics.
* Provides high-level acoustic representations.

---

### MFCC

Mel Frequency Cepstral Coefficients provide traditional acoustic information.

Output:

* 80-dimensional feature vector

Purpose:

* Captures spectral structure.
* Complements learned embeddings.

---

### ECAPA Features

ECAPA-TDNN inspired speaker embeddings are extracted from bark audio.

Output:

* 192-dimensional embedding

Purpose:

* Captures identity-related temporal characteristics.

---

## Fusion3 Architecture

The three feature streams are concatenated:

```text
Dog2Vec : 768
MFCC    : 80
ECAPA   : 192
----------------
Total   : 1040 dimensions
```

The concatenated vector is projected into a compact fingerprint space.

Output:

```text
256-D fingerprint embedding
```

---

## Metric Learning

### Why Metric Learning?

Traditional classification learns:

```text
Input → Class Label
```

Fingerprinting instead learns:

```text
Input → Embedding Space
```

where samples from the same dog are close together and samples from different dogs are far apart.

---

### Triplet Loss

Triplet loss is the final training objective.

Each training step uses:

```text
Anchor Bark
Positive Bark (same dog)
Negative Bark (different dog)
```

Optimization objective:

```text
Distance(anchor, positive)
<
Distance(anchor, negative)
```

This directly trains a fingerprint space suitable for open-set identification.

---

## Enrollment

Each enrolled dog is represented by a profile created from:

* 1 bark
* 3 barks
* 5 barks

Profile construction:

```text
Average Embedding
       ↓
L2 Normalize
       ↓
Dog Profile
```

---

## Identification

For a query bark:

1. Extract fingerprint embedding.
2. Compare against all enrolled profiles.
3. Compute cosine similarity.
4. Select highest similarity score.
5. Apply threshold.

Decision:

```text
Score > Threshold
    → Known Dog

Score < Threshold
    → Unknown Dog
```

---

## Open-Set Evaluation

The system is evaluated using random open-set splits.

For each split:

1. Select known dogs.
2. Hold out unknown dogs.
3. Create enrollment profiles.
4. Evaluate identification.
5. Repeat 100 times.

This produces statistically stable estimates.

---

# Results

## Model Ablation

| Model                      | EER (%)   | DIR@FAR=0.1 (%) |
| -------------------------- | --------- | --------------- |
| DogECAPA-LDE               | 50.00     | 0.00            |
| DogECAPA-Tiny Softmax      | 49.01     | 1.98            |
| DogECAPA-Tiny AAM          | 49.60     | 2.29            |
| Fusion3 + Center Loss      | 27.87     | 56.85           |
| **Fusion3 + Triplet Loss** | **19.49** | **64.06**       |

Triplet Loss produced the best fingerprinting performance.

---

## Primary Benchmark

145 Known Dogs / 10 Unknown Dogs

100 Random Splits

5-Bark Enrollment

| Metric             | Result         |
| ------------------ | -------------- |
| Threshold          | 0.65           |
| Overall Accuracy   | 69.76 ± 3.77%  |
| Known Dog Accuracy | 72.07 ± 0.61%  |
| Unknown Recall     | 46.48 ± 18.41% |
| False Unknown Rate | 4.46 ± 0.39%   |
| EER                | 16.87 ± 0.63%  |
| DIR@FAR=0.001      | 39.96 ± 14.59% |
| DIR@FAR=0.01       | 61.55 ± 12.45% |
| DIR@FAR=0.1        | 88.42 ± 3.54%  |

---

## Open-Set Scaling Study

### 145 Known / 10 Unknown

| Metric  | Value  |
| ------- | ------ |
| EER     | 16.87% |
| DIR@0.1 | 88.42% |

### 100 Known / 55 Unknown

| Metric  | Value  |
| ------- | ------ |
| EER     | 13.86% |
| DIR@0.1 | 87.91% |

### 55 Known / 100 Unknown

| Metric  | Value  |
| ------- | ------ |
| EER     | 12.76% |
| DIR@0.1 | 87.00% |

As the number of unknown dogs increases, the optimal threshold becomes more conservative:

```text
145/10   → 0.65
100/55   → 0.85
55/100   → 0.90
```

while maintaining strong open-set identification performance.

---

## Comparison to Human Speaker Recognition

This work adopts evaluation protocols from open-set speaker identification systems such as VoxBlink.

| Metric        | VoxBlink (Human Speech) | Ours (Dog Barking) |
| ------------- | ----------------------- | ------------------ |
| Enrollment    | 5 utterances            | 5 barks            |
| DIR@FAR=0.001 | 24.94%                  | 39.96%             |
| DIR@FAR=0.01  | 87.12%                  | 61.55%             |
| DIR@FAR=0.1   | 94.34%                  | 88.42%             |

Although human speech remains easier to identify, the results demonstrate that biometric-style fingerprinting concepts can successfully transfer to canine vocalizations.

---

## Repository Structure

```text
scripts/

01_prepare_dataset.py
02_prepare_fusion_data.py
03_extract_ecapa.py

65_train_triplet_fusion3.py
67_eval_triplet_fusion3.py

68_make_final_model_table.py
70_make_ablation_table.py

71_triplet_fusion3_100_splits.py
76_triplet_fusion3_100_unknown_dogs.py
77_triplet_fusion3_55_unknown_dogs.py

78_full_sanity_check.py
79_plot_score_distributions.py
```

---

## Main Contributions

1. First large-scale open-set dog bark fingerprinting study using metric learning.
2. Fusion of Dog2Vec, MFCC, and ECAPA representations.
3. Triplet-loss fingerprint embedding space.
4. Evaluation across 100 random open-set splits.
5. Comparison with human speaker recognition evaluation protocols.
6. Demonstration that dog identity can be recovered from bark vocalizations using biometric-style methods.

---

## Future Work

* Larger enrollment populations
* Additional dog identities
* CAM++ and ECAPA2 architectures
* Self-supervised dog vocalization pretraining
* Audio-visual fingerprinting using bark localization and video cues
* Real-world multi-dog household deployment
