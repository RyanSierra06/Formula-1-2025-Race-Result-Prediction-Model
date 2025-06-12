import time
from urllib.request import urlopen
from urllib.parse import quote
from urllib.error import HTTPError
import json
import pandas as pd

def safe_urlopen(url, retries=3, backoff=1.0):
    for i in range(retries):
        try:
            return urlopen(url)
        except HTTPError as e:
            if e.code == 429 and i < retries - 1:
                time.sleep(backoff * (2**i))
                continue
            else:
                raise

def get_race_session_keys(country, location ,year):
    sessions = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Sprint", "Race"]
    session_keys = {}

    for session in sessions:
        url = (
            f"https://api.openf1.org/v1/sessions?"
            f"country_name={quote(country)}&"
            f"location={quote(location)}&"
            f"session_name={quote(session)}&"
            f"year={year}"
        )
        try:
            response = safe_urlopen(url)
            data = json.loads(response.read().decode("utf-8"))
            if data:
                session_keys[session] = {
                    "session_name": session,
                    "session_key": data[0]["session_key"],
                    "meeting_key": data[0]["meeting_key"],
                }
            else:
                # print(f"No session keys found for {session} in {country} during {year}")
                session_keys[session] = None
        except Exception as e:
            # print(f"Error fetching {session}: {e}")
            session_keys[session] = None

    return session_keys


def get_all_races_in_year(year):
    response = safe_urlopen(f'https://api.openf1.org/v1/meetings?year={year}')
    data = json.loads(response.read().decode('utf-8'))
    races = []
    for meeting in data:
        races.append({
            "country": meeting["country_name"],
            "location": meeting["location"],
            "official_name": meeting["meeting_name"],
            "date": meeting["date_start"]
        })
    return sorted(races, key=lambda x: x['date'])


def get_all_drivers_in_session(session_key):
    response = urlopen(f'https://api.openf1.org/v1/drivers?session_key={session_key}')
    data = json.loads(response.read().decode('utf-8'))
    drivers = []
    for driver in data:
        drivers.append(driver["first_name"] + " " + driver["last_name"])
    return drivers


def get_lap_info(session_key):
    drivers_url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
    try:
        drivers_response = safe_urlopen(drivers_url)
        drivers_data = json.loads(drivers_response.read().decode("utf-8"))
    except HTTPError as e:
        # print(f"Error fetching drivers for session {session_key}: {e}")
        return pd.DataFrame()

    all_laps = []
    if not drivers_data:
        # print("No Driver Data Found For Session: " + session_key)
        return pd.DataFrame()

    for driver in drivers_data:
        driver_number = driver["driver_number"]
        laps_url = f"https://api.openf1.org/v1/laps?session_key={session_key}&driver_number={driver_number}"
        try:
            response = safe_urlopen(laps_url)
            lap_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            # print(f"Error fetching laps for driver {driver_number} in session {session_key}: {e}")
            continue

        if lap_data:
            for lap in lap_data:
                lap["driver_name"] = f"{driver['first_name']} {driver['last_name']}"
                lap["team_name"] = driver["team_name"]
            all_laps.extend(lap_data)

    if not all_laps:
        # print("No Laps Found In Lap Data Found For Session:" + session_key)
        return pd.DataFrame()

    df = pd.DataFrame(all_laps)
    df = df.sort_values(["driver_number", "lap_number"])
    return df[["driver_name", "team_name", "driver_number", "lap_number", "duration_sector_1", "duration_sector_2",
               "duration_sector_3", "i1_speed", "i2_speed", "lap_duration", "st_speed", "is_pit_out_lap",]]


def get_race_results(session_key):
    position_url = f"https://api.openf1.org/v1/position?session_key={session_key}"
    try:
        response = safe_urlopen(position_url)
        position_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        # print(f"Error fetching race positions for session {session_key}: {e}")
        return pd.DataFrame()

    if not position_data:
        # print(f"No position data found for session {session_key}")
        return pd.DataFrame()

    final_positions = {}
    for entry in position_data:
        driver = entry["driver_number"]
        timestamp = entry["date"]
        if driver not in final_positions or timestamp > final_positions[driver]["date"]:
            final_positions[driver] = entry

    drivers_url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
    try:
        response = safe_urlopen(drivers_url)
        drivers_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        # print(f"Error fetching driver info for session {session_key}: {e}")
        return pd.DataFrame()

    driver_map = {int(d["driver_number"]): d for d in drivers_data}

    rows = []
    for driver_number, entry in final_positions.items():
        driver_info = driver_map.get(int(driver_number))
        if driver_info:
            rows.append({
                "driver_number": driver_number,
                "driver_name": f"{driver_info['first_name']} {driver_info['last_name']}",
                "team_name": driver_info["team_name"],
                "Race_position": entry["position"]
            })

    return pd.DataFrame(rows)


def get_best_avg_lap_times(session_key, session_name):
    if session_name == "Race":
        return get_race_results(session_key)

    df = get_lap_info(session_key)
    if df.empty:
        # print(f"No lap data found for session: {session_key}")
        return pd.DataFrame()

    if "is_pit_out_lap" in df.columns:
        df = df[df["is_pit_out_lap"] == False]

    grouped = df.groupby(["driver_number", "driver_name", "team_name"])

    result = grouped["lap_duration"].agg(["min", "mean"]).rename(
        columns={"min": f"{session_name}_best_lap", "mean": f"{session_name}_avg_lap"}
    ).reset_index()

    sectors = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
    for i, sector in enumerate(sectors, start=1):
        if sector in df.columns:
            sector_stats = grouped[sector].agg(["min", "mean"]).rename(
                columns={
                    "min": f"{session_name}_best_sector_{i}",
                    "mean": f"{session_name}_avg_sector_{i}",
                }
            )
            result = result.merge(
                sector_stats,
                on=["driver_number", "driver_name", "team_name"],
                how="left",
            )

    result = result.dropna(subset=[f"{session_name}_best_lap"])

    if not result.empty:
        result = result.sort_values(f"{session_name}_best_lap")
        result[f"{session_name}_position"] = (
            result[f"{session_name}_best_lap"]
            .rank(method="dense", ascending=True)
            .astype(int)
        )
        cols = [f"{session_name}_position"] + [
            c for c in result.columns if c != f"{session_name}_position"
        ]
        result = result[cols]

    return result