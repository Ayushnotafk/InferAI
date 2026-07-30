# Calibration Study

Prefer **isotonic** calibration: hybrid ECE=0.3264, hybrid accuracy=0.7600. Production API remains unchanged; apply calibrator offline if adopted.

| Method | ML Acc | ML ECE | Hybrid Acc | Hybrid ECE |
|--------|--------|--------|------------|------------|
| none | 0.600 | 0.142 | 0.765 | 0.346 |
| temperature | 0.600 | 0.161 | 0.765 | 0.343 |
| platt | 0.750 | 0.196 | 0.770 | 0.383 |
| isotonic | 0.700 | 0.103 | 0.760 | 0.326 |
