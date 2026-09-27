# Phase-3 split (matcher_train / dev_eval / calibration_holdout)

Built strictly inside validation_v1's `development` partition (1,765,456 entities); `validation` untouched.

Seed: 42. Fractions: {'matcher_train': 0.7, 'dev_eval': 0.15, 'calibration_holdout': 0.15}.

Counts: {'matcher_train': 1235818, 'calibration_holdout': 264819, 'dev_eval': 264819}

Manifest SHA-256 (order-independent): `32d16a8436af04fed3e2d0e81058da96c26d4ef8d8e670f109cc18bb3e66fa94`

Stratification key: (country, singleton, match_bucket), same as validation_v1. 6 strata.
