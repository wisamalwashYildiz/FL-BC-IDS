# FL-BC-IDS Non-IID and Participation Stress — Publication-Corrected

Contract: 2 RSUs, 10 vehicles/RSU, 8 federated rounds, 10 local boosting
rounds/participant/round, seed 42. The heterogeneous training partition covers
all 3,944,668 rows exactly once; validation and test are unchanged.

Client sizes: 60,353–433,518 (7.183x). Attack fractions:
0.0087919165–0.8702817564.

Combined participation stress yields 142 successful fits and 18 unavailable
contributions out of 160 planned. The schedule includes persistent dropout,
an isolated missed round, and two multi-round burst outages at each RSU. One
stale-round signed report and one cross-RSU-scope report are deliberately
submitted and both are rejected.

This is contribution-level FL stress, not packet-level mobility/radio/handoff
validation. The three scenarios are single-seed diagnostics; their utility
differences must not be interpreted causally as showing that dropout improves
accuracy.

Publication repairs change no retained scientific result. Machine-local artifact
paths were replaced by logical archive locators. The Python publication copy was
hardened for code/core repository layout and stripped of all contentReference
comment remnants.
