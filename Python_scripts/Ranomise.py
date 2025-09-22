import numpy as np
import pandas as pd
from itertools import product
from math import ceil

def stratified_randomisation(n_subjects: int,
                             group_ratio,
                             strata_probs: dict | None = None,
                             *,
                             seed: int | None = None):
    """
    Stratified block randomisation with fixed quotas per stratum.

    Parameters
    ----------
    n_subjects   : int
        Общий размер выборки.
    group_ratio  : list/tuple/array
        Например [1, 1] или [2, 1] или [0.4, 0.6].
        Если сумма = 1 → трактуется как вероятности,
        иначе — как целочисленное соотношение.
    strata_probs : dict | None
        Иерархия вида
            {"Sex": {"Male": .5, "Female": .5},
             "Age": {"<40": .6, "≥40": .4}}
        Если None или {}, стратификации нет.
    seed         : int | None
        Фиксация генератора случайных чисел.

    Returns
    -------
    assignments : pd.DataFrame  (по строке на испытуемого)
    summary     : dict          (разрезы по группам/стратам/комбинациям)

    Example
    -------
    ```
    strata = {
        "Sex":        {"Male": .5,  "Female": .5},
        "Age group":  {"18-39": 1/3, "40-65": 1/3, "65+": 1/3}
    }
    
    assignments, info = stratified_randomisation(
                           n_subjects = 300,
                           group_ratio= [1, 1],
                           strata_probs={},
                            # strata_probs=strata,
                           seed=2024)
    ```    
    """
    rng = np.random.default_rng(seed)

    # ----------------- блок 1. группы -------------------------------------
    ratio = np.asarray(group_ratio, dtype=float)
    if np.isclose(ratio.sum(), 1.0):
        ratio = ratio / np.min(ratio)          # превратили в целые
    n_groups   = len(ratio)
    grp_labels = [f"Group_{i+1}" for i in range(n_groups)]
    blocksize  = int(ratio.sum())

    # ----------------- блок 2. сетка страт -------------------------------
    if not strata_probs:                       # None или {}
        strata_probs = {"_dummy": {"_all": 1.0}}

    strata_cols   = list(strata_probs.keys())
    strata_levels = [list(d.keys()) for d in strata_probs.values()]

    strata_df = pd.DataFrame(list(product(*strata_levels)),
                             columns=strata_cols)

    # вероятность каждой комбинации
    prob = np.ones(len(strata_df), dtype=float)
    for col in strata_cols:
        prob *= strata_df[col].map(strata_probs[col]).values
    prob /= prob.sum()                         # на всякий случай нормируем
    strata_df["prob"] = prob

    # ----------- точные квоты (Largest-Remainder) ------------------------
    exact_raw = n_subjects * prob
    base      = np.floor(exact_raw).astype(int)
    remainder = exact_raw - base
    missing   = n_subjects - base.sum()
    if missing:
        base[np.argsort(-remainder)[:missing]] += 1
    strata_df["n"] = base
    assert strata_df["n"].sum() == n_subjects

    # ----------------- блок 3. разворачиваем испытуемых ------------------
    subjects = []
    sid = 1
    for _, row in strata_df.iterrows():
        if row["n"] == 0:
            continue
        block = pd.DataFrame({"Subject": np.arange(sid, sid + row["n"])})
        for c in strata_cols:
            block[c] = row[c]
        subjects.append(block)
        sid += row["n"]

    df = pd.concat(subjects, ignore_index=True)

    # ----------------- блок 4. распределение групп -----------------------
    df["Group"] = None
    for combo, idx in df.groupby(strata_cols, sort=False).groups.items():
        n_combo = len(idx)
        n_blocks = ceil(n_combo / blocksize)
        seq = []
        for _ in range(n_blocks):
            blk = np.concatenate([[lab] * int(rep)
                                   for lab, rep in zip(grp_labels, ratio)])
            rng.shuffle(blk)
            seq.extend(blk)
        df.loc[idx, "Group"] = seq[:n_combo]

    # ----------------- блок 5. сводки ------------------------------------
    grp_counts = (df["Group"]
                  .value_counts()
                  .rename_axis("Group")
                  .reset_index(name="n"))
    str_counts = {s: (df[s]
                      .value_counts()
                      .rename_axis(s)
                      .reset_index(name="n"))
                  for s in strata_cols}

    rand_lists = {combo if isinstance(combo, tuple) else (combo,):   # для 1-страты
                  list(df.loc[idx]
                         .sort_values("Subject")["Group"].values)
                  for combo, idx in df.groupby(strata_cols,
                                               sort=False).groups.items()}

    summary = dict(total_n=n_subjects,
                   group_counts=grp_counts,
                   strata_counts=str_counts,
                   rand_lists=rand_lists)

    return df, summary
