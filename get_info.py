import requests
import datetime as dt
import time
import os
import pandas as pd
import urllib.parse
import json
from datetime import datetime, timezone, timedelta

api_key = "api"


request_header = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": api_key,
}

def get_r(url):
    while True:
        try:
            r = requests.get(url, headers=request_header, timeout=3)
            if r.status_code == 429:  # Rate Limit Exceeded
                print("Rate limit exceeded. Waiting for 10 seconds...")
                time.sleep(10)  
                continue
            break
        except Exception as e:
            print(e, "Retrying in 5 seconds...")
            time.sleep(5)
            continue
    return r

def save_match_ids_to_json(match_ids, filename):
    """
    매치 ID를 JSON 파일로 저장하는 함수.

    Args:
        match_ids (dict): 사용자별 매치 ID.
        filename (str): 저장할 파일 이름.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(match_ids, file, ensure_ascii=False, indent=4)


def get_match_ids(puuid):
    """
    소환사의 모든 매치 ID를 가져옵니다.

    Args:
        puuid (str): 소환사의 PUUID.

    Returns:
        list: 매치 ID 목록 (최신 매치부터 오래된 순으로 정렬됨).
    """
    match_ids = []
    start = 0
    count = 100  # 한 번에 가져올 매치 ID 개수

    while True:
        # 매치 ID 가져오기 요청 URL
        base_url = f"https://asia.api.riotgames.com/tft/match/v1/matches/by-puuid/{puuid}/ids?start={start}&count={count}"
        response = get_r(base_url)

        if response.status_code == 200:
            new_matches = response.json()
            if not new_matches:  # 더 이상 가져올 매치가 없으면 종료
                break
            match_ids.extend(new_matches)
            start += count  # 다음 요청의 시작점을 증가
        else:
            print(f"Error fetching match IDs: {response.status_code}, {response.text}")
            break

    return match_ids

def get_match_info(match_ids):
    match_info = {}

    for match_id in match_ids:
        # 매치 ID 가져오기 요청 URL
        base_url = f"https://asia.api.riotgames.com/tft/match/v1/matches/{match_id}"
        response = get_r(base_url)
    
        if response.status_code == 200:
            match_data = response.json()
            match_info[match_id] = match_data  # match_id를 키로 데이터 저장
        else:
            print(f"Error fetching match details for {match_id}: {response.status_code}, {response.text}")

    return match_info


def get_puid(name, tag):
    print(f"Loading..... \nget {name}'s puuid ")
    match_detail_url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
    response = get_r(match_detail_url)

    if response.status_code != 200:
        print(f"Error fetching match details for {name} # {tag}: {response.status_code}")
        return None

    match_data = response.json()
    puid = match_data.get("puuid", None)

    return puid

def load_match_data(filename):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_data (nametage, json_file_name):
    names = {}
    for name, tag in nametage.items():
        names[name] = get_puid(name,tag)
    all_match_data = {}
    for name in names:
        match_ids = get_match_ids(names[name])

        if match_ids:
                all_match_data[name] = match_ids
                print(f"Saved {len(match_ids)} matches for {name}.")
        else:
            print("No matches found.")

    save_match_ids_to_json(all_match_data,json_file_name)
    print(f"Match data saved to {json_file_name}")


def save_match_info(match_ids, filename):
    info = {}
    for name, ids in match_ids.items():
        info[name] = get_match_info(ids)  # 사용자별 데이터를 저장
    
    save_match_ids_to_json(info, filename)

def save_json_file(data, filename):
    """
    Save data to a JSON file.
    Args:
        data (dict): Data to save.
        filename (str): Path to the JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def filter_by_set_number(data, target_set_number):
    """
    Filter the data for a specific TFT set number.
    Args:
        data (dict): The original JSON data.
        target_set_number (int): The target TFT set number to filter.

    Returns:
        dict: Filtered data.
    """
    filtered_data = {}
    for summoner_name, matches in data.items():
        for match_id, match_data in matches.items():
            if match_data.get("info", {}).get("tft_set_number") == target_set_number:
                if summoner_name not in filtered_data:
                    filtered_data[summoner_name] = {}
                filtered_data[summoner_name][match_id] = match_data
    return filtered_data

def to_korea_time(epoch_ms):
    korea_timezone = timezone(timedelta(hours=9))
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=korea_timezone)
    return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.strftime("%A")

def main():
    nametage = {"name":"tag"}

    for name, tag in nametage.items():
        os.makedirs(f"./data/{name}_output_folder", exist_ok=True)
        data = load_match_data(f"./data/{name}_match_info.json")

        target_set_numbers = [ 11, 12, 13]    

        for set_number in target_set_numbers:

           filtered_data = filter_by_set_number(data, set_number)

           output_file = os.path.join(f"./data/{name}_output_folder", f"tft_set_{set_number}.json")


           save_json_file(filtered_data, output_file)
           print(f"Filtered data for TFT set {set_number} saved to {name}_outout_folder")



main()