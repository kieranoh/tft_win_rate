import pandas as pd
import json
from datetime import datetime, timezone, timedelta

# load fike
file_path = r"./data/game_data.json"

def to_korea_time(epoch_ms):
    korea_timezone = timezone(timedelta(hours=9))
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=korea_timezone)
    return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.strftime("%A")

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

all_data_list = []

# get data from json
for game_key, game_data in data.items():
   
    game_info = game_data["info"]
    participants = game_info["participants"]

    # except if just one participants
    if len(participants) <= 1:
        print(f"special mode, game ID {game_info['gameId']} excepted.")
        continue
    
    # transit time
    game_datetime, game_day = to_korea_time(game_info["game_datetime"])
    
    
    for p in participants:
        
        # Traits  (tier_current > 0)
        traits = [
            f"{trait['name']} (Level {trait['tier_current']})"
            for trait in p["traits"]
            if trait["tier_current"] > 0
        ]
        
        # Units 
        units = [
            {
                "unit_id": unit["character_id"],
                "tier": unit["tier"],
                "items": ", ".join(unit["itemNames"])
            }
            for unit in p["units"]
        ]
        units_str = "; ".join(
            f"{unit['unit_id']} (Tier {unit['tier']}, Items: {unit['items']})" for unit in units
        )
        
        # save data
        all_data_list.append({
            "game ID": game_info["gameId"],
            "generated time": game_datetime,
            "game day": game_day,
            "name": p["riotIdGameName"],
            "placement": p["placement"],
            "gold left": p["gold_left"],
            "level": p["level"],
            "time elimented": p["time_eliminated"],
            "total damage": p["total_damage_to_players"],
            "Traits": ", ".join(traits),
            "Units": units_str,
        })

# make dataframe
all_data_df = pd.DataFrame(all_data_list)
print(len(all_data_df))
print(all_data_df.duplicated().sum())

# remove duplicate
all_data_df.drop_duplicates(inplace=True)
print(len(all_data_df))

# remove TFT  
def remove_tft_keywords(df):
    # target
    replacements = [f"TFT{i}_" for i in range(15)] + ["TFT_Item_", "Augment_", "TFT"]
    
    for r in replacements:
        df = df.applymap(lambda x: x.replace(r, "") if isinstance(x, str) else x)
    return df

# cleanup
all_data_df = remove_tft_keywords(all_data_df)

# order by time generated
all_data_df["generated time"] = pd.to_datetime(all_data_df["generated time"], errors="coerce")
# print(all_data_df["generated time"].head())
# print(all_data_df["generated time"].dtype)
all_data_df["generated time"] = pd.to_datetime(all_data_df["generated time"])
all_data_df.sort_values(by="generated time", ascending=False, inplace=True)
all_data_df["generated time"] = all_data_df["generated time"].dt.strftime("%Y-%m-%d %H:%M:%S")

# save the data
output_path = './data/game_data.xlsx'
with pd.ExcelWriter(output_path) as writer:
    all_data_df.to_excel(writer, sheet_name='game data', index=False)

print(f"Cleaned data saved to {output_path}")
