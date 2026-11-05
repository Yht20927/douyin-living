# Clipping Pipeline Baseline

- Generated: 2026-11-05T10:29:07+00:00
- Git commit: `626eae0`
- Python: 3.14.4
- Platform: Linux-7.0.0-29-generic-x86_64-with-glibc2.43
- Sensitivity: 2.5
- Duration bounds: 15.0–90.0s

## Sample rooms

| Room | Status | Duration (s) | Clips | Elapsed (s) | Drop reasons |
| --- | --- | ---: | ---: | ---: | --- |
| 887951935002 | ok | 20 | 0 | 0.003 | - |

## Synthetic one-hour recall

| Expected peaks | Clips | Recall | Elapsed (s) |
| ---: | ---: | ---: | ---: |
| 60 | 60 | 100.0% | 0.058 |

## Alignment

### 887951935002

| Signal | Old len | New len | Method |
| --- | ---: | ---: | --- |
| dmDensity | 17 | 20 | interpolate |
| dmAcceleration | 17 | 20 | interpolate |
| dmEntropy | 17 | 20 | interpolate |
| dmSentiment | 17 | 20 | interpolate |
| dmUniqueUsers | 17 | 20 | interpolate |
| dmUtr | 17 | 20 | interpolate |
| asrKeyword | 17 | 20 | pad_zeros |
| topicChange | 17 | 20 | pad_zeros |
| speakerChange | 17 | 20 | pad_zeros |


## Coverage

- Total: **76.6%** (3216 statements, 752 missing)

## Notes

- Generated from checked-in feature artifacts; no model downloads are required.
- A sample room with zero clips means the scorer found no candidate above the
  current threshold, not that the pipeline crashed.
- Re-run this script after changing scorer thresholds or weights.
