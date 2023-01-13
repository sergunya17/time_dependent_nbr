# Time-dependent Next-basket Recommendations

## Hyperparameter search space

### G-Pop (top_popular)
```python
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary", "log"])
```

### GP-Pop (top_personal)
```python
min_freq = trial.suggest_int("min_freq", 1, 20)
preprocessing_popular = trial.suggest_categorical(
    "preprocessing_popular", [None, "binary", "log"]
)
preprocessing_personal = trial.suggest_categorical(
    "preprocessing_personal", [None, "binary", "log"]
)
```

### UP-CF@r (up_cf)
```python
recency = trial.suggest_categorical("recency", [1, 5, 25, 100])
q = trial.suggest_categorical("q", [1, 5, 10, 50, 100, 1000])
alpha = trial.suggest_categorical("alpha", [0, 0.25, 0.5, 0.75, 1])
topk_neighbors = trial.suggest_categorical("topk_neighbors", [None, 10, 100, 300, 600, 900])
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary"])
```

### TIFU-KNN (tifuknn)
```python
num_nearest_neighbors = trial.suggest_categorical(
    "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
)
within_decay_rate = trial.suggest_categorical(
    "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
group_decay_rate = trial.suggest_categorical(
    "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
group_count = trial.suggest_int("group_count", 2, 23)
```

### TIFU-KNN-TA (tifuknn_time_days)
```python
num_nearest_neighbors = trial.suggest_categorical(
    "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
)
within_decay_rate = trial.suggest_categorical(
    "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
group_decay_rate = trial.suggest_categorical(
    "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
group_size_days = trial.suggest_int("group_size_days", 1, 365)
use_log = trial.suggest_categorical("use_log", [True, False])
```

### TIFU-KNN-TD (tifuknn_time_days_next_ts)
```python
num_nearest_neighbors = trial.suggest_categorical(
    "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
)
within_decay_rate = trial.suggest_categorical(
    "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
group_decay_rate = trial.suggest_categorical(
    "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
group_size_days = trial.suggest_int("group_size_days", 1, 365)
use_log = trial.suggest_categorical("use_log", [True, False])
```


