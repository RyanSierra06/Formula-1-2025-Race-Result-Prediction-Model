import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DATA_DIR = "Formula_1_Grandprix_Data"

def featurize_data(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Drop entries with missing race position
    if "Race_position" in df:
        df = df.dropna(subset=["Race_position"]).reset_index(drop=True)

    keep_cols = ["driver_number", "driver_name", "team_name", "Race_position"]
    practice_sessions = ["Practice 1", "Practice 2", "Practice 3"]
    other_sessions = ["Qualifying", "Sprint"]

    # 1) Basic features: participation flag, rel_to_fastest, consistency
    for session in practice_sessions + other_sessions:
        best_col = f"{session}_best_lap"
        avg_col = f"{session}_avg_lap"
        pos_col = f"{session}_position"

        # did the session?
        if best_col in df:
            df[f"did_{session.replace(' ', '_')}"] = df[best_col].notna().astype(int)
        else:
            df[f"did_{session.replace(' ', '_')}"] = 0

        # relative to fastest
        if best_col in df:
            fastest = df[best_col].min()
            df[f"{session}_rel_to_fastest"] = df[best_col] / fastest

        # consistency = avg - best
        if best_col in df and avg_col in df:
            df[f"{session}_consistency"] = df[avg_col] - df[best_col]

    # 2) Improvement from Practice 3 to Qualifying
    if "Practice 3_position" in df and "Qualifying_position" in df:
        df["Quali_vs_P3_pos_delta"] = df["Practice 3_position"] - df["Qualifying_position"]

    # 3) Deltas: Qualifying vs Practices
    for p in practice_sessions:
        p_best = f"{p}_best_lap"
        p_avg = f"{p}_avg_lap"
        if p_best in df and "Qualifying_best_lap" in df:
            df[f"Quali_minus_{p.replace(' ', '_')}_best"] = df["Qualifying_best_lap"] - df[p_best]
        if p_avg in df and "Qualifying_avg_lap" in df:
            df[f"Quali_minus_{p.replace(' ', '_')}_avg"] = df["Qualifying_avg_lap"] - df[p_avg]

    # 4) Sprint vs Qualifying deltas
    if "Sprint_best_lap" in df and "Qualifying_best_lap" in df:
        df["Sprint_minus_Quali_best"] = df["Sprint_best_lap"] - df["Qualifying_best_lap"]
    if "Sprint_avg_lap" in df and "Qualifying_avg_lap" in df:
        df["Sprint_minus_Quali_avg"] = df["Sprint_avg_lap"] - df["Qualifying_avg_lap"]

    # 5) Practice-to-practice deltas (P2-P1, P3-P2)
    for (a, b) in [("Practice 1", "Practice 2"), ("Practice 2", "Practice 3")]:
        a_best, b_best = f"{a}_best_lap", f"{b}_best_lap"
        a_avg, b_avg = f"{a}_avg_lap", f"{b}_avg_lap"
        col_suffix = f"{b.replace(' ', '_')}_minus_{a.replace(' ', '_')}"
        if a_best in df and b_best in df:
            df[f"{col_suffix}_best"] = df[b_best] - df[a_best]
        if a_avg in df and b_avg in df:
            df[f"{col_suffix}_avg"] = df[b_avg] - df[a_avg]

    # 6) Sector aggregates across practices
    for i in [1, 2, 3]:
        sector_cols = [f"{s}_best_sector_{i}" for s in practice_sessions if f"{s}_best_sector_{i}" in df]
        if sector_cols:
            df[f"best_sector_{i}"] = df[sector_cols].min(axis=1)
        sector_avg_cols = [f"{s}_avg_sector_{i}" for s in practice_sessions if f"{s}_avg_sector_{i}" in df]
        if sector_avg_cols:
            df[f"avg_sector_{i}"] = df[sector_avg_cols].mean(axis=1)

    # 7) vs-team normalization
    if "team_name" in df:
        for session in practice_sessions + ["Qualifying"]:
            col = f"{session}_best_lap"
            if col in df:
                medians = df.groupby("team_name")[col].transform("median")
                df[f"{col}_vs_team"] = df[col] / medians

    # Finalize
    y = df["Race_position"] if "Race_position" in df else None
    feature_cols = [c for c in df.columns if c not in keep_cols]
    X = df[feature_cols].copy()
    identifiers = df[["driver_number", "driver_name", "team_name"]]
    return X, y, identifiers


def load_grandprix_df(country, location, year):
    year_dir = os.path.join(BASE_DATA_DIR, f"{year}_data")
    filename = f"{country}_{location}_{year}_grandprix_results_data.csv"
    path = os.path.join(year_dir, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def build_model(target_country, target_location, target_year=2025):
    # Assemble training race list
    train_races = []
    for year in [2023, 2024]:
        for fname in os.listdir(os.path.join(BASE_DATA_DIR, f"{year}_data")):
            country, location = fname.split("_")[:2]
            train_races.append((country, location, year))

    # Include prior races in current year
    for fname in sorted(os.listdir(os.path.join(BASE_DATA_DIR, f"{target_year}_data"))):
        country, location = fname.split("_")[:2]
        if country == target_country and location == target_location:
            break
        train_races.append((country, location, target_year))

    # Featurize training data
    X_list, y_list = [], []
    all_cols = set()
    print(f"Training on {len(train_races)} races...")
    for country, location, year in train_races:
        try:
            df_raw = load_grandprix_df(country, location, year)
            X_race, y_race, _ = featurize_data(df_raw)
            if y_race is None or y_race.isna().all():
                continue
            all_cols |= set(X_race.columns)
            X_list.append(X_race)
            y_list.append(y_race)
        except Exception as e:
            print(f"Skipping {country} {location} {year}: {e}")

    if not X_list:
        print("No training data available")
        return None

    # Align and impute training features
    for i, Xr in enumerate(X_list):
        for col in all_cols - set(Xr.columns):
            Xr[col] = np.nan
    X_train = pd.concat(X_list, ignore_index=True)
    y_train = pd.concat(y_list, ignore_index=True)
    medians = X_train.median()
    X_train = X_train.fillna(medians)

    # Prepare test data
    df_test = load_grandprix_df(target_country, target_location, target_year)
    X_test, y_test, identifiers = featurize_data(df_test)
    for col in all_cols - set(X_test.columns):
        X_test[col] = np.nan
    X_test = X_test[X_train.columns].fillna(medians)

    # Train and predict
    rf = RandomForestRegressor(n_estimators=500, max_depth=8,
                               min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model MAE: {mae:.3f}")
    print(f"Model R2 : {r2:.3f}")

    # Prepare result DataFrame
    results = identifiers.copy()
    results['predicted_pos'] = y_pred
    results['predicted_order'] = results['predicted_pos'].rank(method='dense').astype(int)
    results['actual_pos'] = y_test.values

    return results.sort_values('predicted_order'), mae, r2
