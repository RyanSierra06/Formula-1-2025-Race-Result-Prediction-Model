import os
import pandas as pd
from get_race_data import get_race_session_keys, get_best_avg_lap_times

def build_grandprix_results_dataframe(country, location, year):
    session_keys = get_race_session_keys(country, location, year)
    target_sessions = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Sprint", "Race"]

    output_dir = f"Formula_1_Grandprix_Data"
    os.makedirs(output_dir, exist_ok=True)

    sub_output_dir = f"{output_dir}/{year}_data"
    os.makedirs(sub_output_dir, exist_ok=True)

    grandprix_df = None

    for session_name in target_sessions:
        info = session_keys.get(session_name)
        if info and info["session_key"]:
            session_key = info["session_key"]
            results_data = get_best_avg_lap_times(session_key, session_name)

            if not results_data.empty:
                if grandprix_df is None:
                    grandprix_df = results_data.copy()
                else:
                    grandprix_df = grandprix_df.merge(
                        results_data,
                        how="outer",
                        on=["driver_number", "driver_name", "team_name"],
                        validate="one_to_one",
                    )
            else:
                print(f"No data available for {session_name} in {country} {year}")
    if grandprix_df is None:
        print("No session data was fetched for any session")
        grandprix_df = pd.DataFrame()

    desired_order = ["driver_number", "driver_name", "team_name"]

    for session in target_sessions:
        col_pos = f"{session}_position"
        if col_pos in grandprix_df.columns:
            desired_order.append(col_pos)

        if session != "Race":
            for suffix in [
                "best_lap", "avg_lap",
                "best_sector_1", "avg_sector_1",
                "best_sector_2", "avg_sector_2",
                "best_sector_3", "avg_sector_3",
            ]:
                col = f"{session}_{suffix}"
                if col in grandprix_df.columns:
                    desired_order.append(col)

    grandprix_df = grandprix_df[desired_order]

    file_name = f"{country}_{location}_{year}_grandprix_results_data.csv"
    file_path = os.path.join(sub_output_dir, file_name)
    grandprix_df.to_csv(file_path, index=False)
    print(f"Saved {country} {year} grandprix data to {file_name}")

    return grandprix_df
