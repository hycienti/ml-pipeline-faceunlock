# Flood simulation: latency with different numbers of Docker containers

Locust, 60 simulated users (spawn rate 10/s), 60 s per setup, statistics reset after
the ramp-up. Each container is limited to 2 CPUs and runs the full app and model; nginx balances by
least connections. Every user logs in once, then predicts a registered face (weight 4), predicts a
stranger (2), logs in again (1) and loads the status API (1). Times are milliseconds.

| Containers | Requests | Failures | Requests/s | Median | 95th pct | 99th pct | Predict median | Predict 95th pct |
|---|---|---|---|---|---|---|---|---|
| 1 | 642 | 0 | 12.2 | 3900 | 8600 | 9300 | 3900 | 8500 |
| 2 | 1409 | 0 | 26.0 | 1800 | 3100 | 4100 | 1800 | 3100 |
| 3 | 3282 | 0 | 60.7 | 470 | 1300 | 1800 | 500 | 1300 |
