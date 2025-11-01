"""Football match data fetcher"""
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

class FootballDataOrgFetcher:
    """Fetch from football-data.org API"""
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        self.headers = {"X-Auth-Token": self.api_key}
    
    def get_matches(self, competition_code: str = "PL", season_year: int = 2024,
                   date_from: str = None, date_to: str = None) -> pd.DataFrame:
        """Fetch matches for competition and season"""
        params = {"season": season_year}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        url = f"{self.BASE_URL}/competitions/{competition_code}/matches"
        print(f"Fetching {competition_code} {season_year}/{season_year+1}...")
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 429:
            print("Rate limit. Waiting 60s...")
            time.sleep(60)
            response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        data = response.json()
        matches = []
        for match in data.get("matches", []):
            if match["status"] != "FINISHED":
                continue
            matches.append({
                "date": match["utcDate"][:10],
                "competition": match["competition"]["name"],
                "season": match["season"]["startDate"][:4],
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_goals": match["score"]["fullTime"]["home"],
                "away_goals": match["score"]["fullTime"]["away"],
                "result": self._determine_result(
                    match["score"]["fullTime"]["home"],
                    match["score"]["fullTime"]["away"]
                )
            })
        df = pd.DataFrame(matches)
        if df.empty:
            print(f"No matches found for {competition_code}")
        else:
            print(f"{len(df)} matches")
        return df
    
    @staticmethod
    def _determine_result(home_goals: int, away_goals: int) -> int:
        """0=away win, 1=draw, 2=home win"""
        if home_goals > away_goals:
            return 2
        elif home_goals < away_goals:
            return 0
        return 1
    
    def fetch_historical_seasons(self, competition_code: str = "PL", num_seasons: int = 3) -> pd.DataFrame:
        """Fetch multiple seasons"""
        current_year = datetime.now().year
        all_matches = []
        for i in range(num_seasons):
            season_year = current_year - i
            try:
                df = self.get_matches(competition_code, season_year)
                all_matches.append(df)
                time.sleep(6)
            except (requests.RequestException, ValueError, KeyError) as e:
                print(f"Error season {season_year}: {e}")
                continue
        if all_matches:
            return pd.concat(all_matches, ignore_index=True)
        return pd.DataFrame()

def fetch_from_csv_url(url: str) -> pd.DataFrame:
    """Load from CSV URL"""
    df = pd.read_csv(url)
    column_mapping = {
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals",
        "HTHG": "home_goals_ht", "HTAG": "away_goals_ht"
    }
    df = df.rename(columns=column_mapping)
    if "result_code" in df.columns:
        result_map = {"H": 2, "D": 1, "A": 0}
        df["result"] = df["result_code"].map(result_map)
    elif "home_goals" in df.columns and "away_goals" in df.columns:
        df["result"] = df.apply(
            lambda r: 2 if r["home_goals"] > r["away_goals"] else (0 if r["home_goals"] < r["away_goals"] else 1),
            axis=1
        )
    return df

def fetch_data(source: str = "football-data.org", competition: str = "PL",
               num_seasons: int = 3, output_file: str = "matches.csv") -> pd.DataFrame:
    """Main fetch function"""
    df = pd.DataFrame()
    try:
        if source == "football-data.org":
            print("Fetching from football-data.org...")
            fetcher = FootballDataOrgFetcher()
            df = fetcher.fetch_historical_seasons(competition, num_seasons)
        elif source == "csv":
            csv_url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
            df = fetch_from_csv_url(csv_url)
        if not df.empty:
            output_path = RAW_DIR / output_file
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} matches to {output_path}")
        return df
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("Football Data Fetcher")
    if os.getenv("FOOTBALL_DATA_API_KEY"):
        print("Using football-data.org API")
        df = fetch_data(source="football-data.org", competition="PL", num_seasons=3)
    else:
        print("No API key. Use CSV source.")
        seasons = ["2324", "2223", "2122"]
        all_data = []
        for season in seasons:
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
                df_season = fetch_from_csv_url(url)
                df_season["season"] = f"20{season[:2]}/{season[2:]}"
                all_data.append(df_season)
                time.sleep(1)
            except Exception as e:
                print(f"Failed: {e}")
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            output_path = RAW_DIR / "matches.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} matches")
