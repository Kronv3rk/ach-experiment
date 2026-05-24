# results/

Здесь хранятся артефакты экспериментов в формате JSON.

## Что должно лежать здесь

После полного прогона:

```
results/
├── all_results.json           # Сводные агрегаты по всем сериям
├── effect_size.json           # Дельта Клиффа ACH vs baseline
├── C1/
│   ├── run_000.json … run_029.json
│   └── summary.json
├── C2/
│   ├── run_000.json … run_029.json
│   └── summary.json
├── C3/ … так же
├── C4a/ … так же
├── C4b/ … так же
├── C4c/ … так же
├── C5/
│   └── sensitivity.json
└── plots/
    ├── D_traces_C1.svg, M_traces_C1.svg, …
    ├── boxplot_C1.svg, ecdf_C1.svg, …
```

## Структура одного `run_NNN.json`

Один прогон содержит:

```json
{
  "run_id":  0,
  "seed":    42,
  "series":  "C1",
  "elapsed": 9.4,
  "algorithms": {
    "Static-W":      { "D_trace": [...], "M_trace": [...], "agg": {...}, "violations": [...], "n_violations": 0 },
    "Static-Wt":     { ... },
    "Dynamic-R":     { ... },
    "Bounded-Loads": { ... },
    "ACH":           { ... }
  }
}
```

Поле `agg` содержит ключевые метрики (`D_mean`, `D_max`, `M_cum`, `pi_chg`,
`ell_max`), вычисленные после периода прогрева (40 шагов).

Поле `D_trace` (и `M_trace`) — массив длиной T=2000 со значениями метрик
на каждом шаге. Эти траектории используются для построения графиков и
для покадрового анализа динамики.

## Как заполнить эту директорию

Запустить из корня репозитория:

```bash
python scripts/run_all.py
python scripts/build_all_results.py
python scripts/effect_size.py --plots
python scripts/plot_traces.py --input-dir results
```

Время полного прогона — около часа на типовой рабочей станции.
Все результаты детерминированы (фиксированные семена).
