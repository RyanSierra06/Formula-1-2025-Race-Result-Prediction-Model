from build_model import build_model
from get_race_data import get_all_races_in_year

if __name__ == '__main__':
    print("Welcome to the Formula 1 2025 Race Prediction ML Model!")

    races_2025 = get_all_races_in_year(2025)

    if not races_2025:
        print("No 2025 races found in API")
        exit()

    print("\n2025 Races:")
    for i, race in enumerate(races_2025, 1):
        print(f"{i}. {race['country']} - {race['location']}")

    race_country = input("\nEnter Grand Prix Country (e.g. 'Italy'):\n")
    race_location = input("Enter Circuit Location (e.g. 'Imola'):\n")

    print("Building model and predicting results...")
    res_tuple = build_model(race_country, race_location, 2025)
    if res_tuple is None:
        print("Prediction failed. Please check inputs and try again.")
        exit(1)

    results_df, mae, r2 = res_tuple

    print("Predicted Finishing Order:")
    print(results_df[["predicted_order",
                      "driver_number",
                      "driver_name",
                      "predicted_pos",
                      "actual_pos"]]
          .to_string(index=False))
