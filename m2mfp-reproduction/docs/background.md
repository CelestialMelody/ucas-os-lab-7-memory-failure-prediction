# Background

## Memory Error Terms

- CE, Correctable Error: an error corrected by ECC. It does not immediately crash the system, but repeated CEs often indicate a degrading DIMM.
- UE, Uncorrectable Error: an error that ECC cannot correct. UE is the target failure event in this project.
- ECC: error-correcting code used by server memory to detect and correct bit errors.
- DIMM: memory module. In the data, each feather file corresponds to one DIMM serial number.
- Lead time: the gap between the prediction timestamp and the failure time. This project uses 15 minutes.
- Prediction window: the future period considered by a positive label. This project uses 7 days.
- Precision: among predicted failures, the fraction that are true failures.
- Recall: among true failures, the fraction that are detected.
- F1: harmonic mean of precision and recall, used because failures are rare.

## CE Logs

The raw Stage1 logs contain timestamps, DIMM identifiers, memory location fields
such as device/bank/row/column, error type, and retry parity information. The
core assumption is that UE events are often preceded by abnormal CE patterns:
more frequent CEs, localized spatial clusters, or distinctive bit-level parity
patterns.

## M2-MFP Idea

M2-MFP models memory failure prediction through multi-scale, multi-level
features:

- Time scale: recent CE counts and bursts in different windows.
- Space scale: whether errors concentrate in a device, bank, row, column, or cell.
- Bit scale: whether retry parity bits show meaningful DQ/beat structure.

This reproduction implements these ideas as BSFE, time-patch, and time-point
features and evaluates them on SmartMem Stage1 raw logs.
