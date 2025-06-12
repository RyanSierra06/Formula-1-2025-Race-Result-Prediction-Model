from get_race_data import get_all_races_in_year
from build_dataframe import build_grandprix_results_dataframe

def build_all_csv_files():
    all_races = []

    for year in [2023, 2024, 2025]:
        races = get_all_races_in_year(year)
        for race in races:
            race["year"] = year
            all_races.append(race)

    for race in all_races:
        country = race["country"]
        location = race["location"]
        year = race["year"]

        try:
            df_raw = build_grandprix_results_dataframe(country, location, str(year))
            if df_raw.empty:
                print(f"Skipping {country} {year} - no data")
                continue

        except Exception as e:
            print(f"Error processing {country} {location} {year}: {str(e)}")
            continue


if __name__ == "__main__":
    print("Building CSV files....")
    build_all_csv_files()