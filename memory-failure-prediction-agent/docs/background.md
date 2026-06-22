# Background

## Problem

The task is to predict whether a DIMM will experience an uncorrectable memory
failure in a future prediction window. The input is a time series of extracted
memory error features; the output is a candidate failure list with
`serial_number`, `prediction_timestamp`, and `serial_number_type`.

## Key Terms

- CE, Correctable Error: an error corrected by ECC. CE history is the predictive signal.
- UE, Uncorrectable Error: the target failure event.
- DIMM: memory module identified by a serial number.
- Stage2 extracted features: official pre-engineered feature tables derived from
  raw CE logs.
- Threshold search: selecting a probability cutoff that maximizes validation F1.
- Top-k: selecting the highest-risk DIMMs when a fixed number of candidates is needed.
- Per-type modeling: training separate models for type_A and type_B DIMMs.

## Why an Agent Search

Failure labels are rare and data distributions differ by DIMM type. A single
default classifier and threshold may not be stable. The agent search enumerates
models, sampling rules, feature subsets, threshold rules, and per-type variants,
then selects the best local validation strategy.
