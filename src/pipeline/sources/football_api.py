import requests
import pandas as pd
import logging
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)

class FootballAPIScraper:
    def __init__(self, api_key: str = None, data_dir: Path = None):
        self.api_key = api_key or settings.FOOTBALL_DATA_API_KEY
        self.data_dir = data_dir or settings.RAW_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            logger.warning("No API Key provided for Football-Data.org. This source may fail.")
        
        self.headers = {"X-Auth-Token": self.api_key}

    def scrape_premier_league(self, start_year: int = None, end_year: int = None):
        logger.info("Fetching data from Football-Data.org API")
        url = f"{settings.FOOTBALL_API_BASE_URL}/competitions/PL/matches"
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            matches = data.get("matches", [])
            if not matches:
                logger.warning("No matches found from API.")
                return

            logger.info(f"Fetched {len(matches)} matches from API.")
            self._process_and_save(matches)
            
        except Exception as e:
            logger.error(f"API request failed: {e}")

    def _process_and_save(self, matches_data: list):
        """Converts API JSON format to the project's CSV schema."""
        processed_rows = []
        stats_placeholders = ["xg", "xga", "poss", "sh", "sot", "dist", "fk", "pk", "pkatt"]
        
        for m in matches_data:
            match_date = pd.to_datetime(m["utcDate"])
            score = m["score"]["fullTime"]
            
            row = {
                "date": match_date.strftime("%Y-%m-%d"),
                "time": match_date.strftime("%H:%M"),
                "comp": "Premier League",
                "round": f"Matchweek {m.get('matchday', 0)}",
                "day": match_date.strftime("%a"),
                "venue": "Home" if m["homeTeam"]["name"] else "Away",
                "result": self._get_result(m["score"]),
                "gf": score["home"],
                "ga": score["away"],
                "opponent": m["awayTeam"]["name"],
                "team": m["homeTeam"]["name"],
                "season": m["season"]["startDate"][:4],
                "status": m["status"]
            }
            
            for stat in stats_placeholders:
                row[stat] = 0.0
                
            processed_rows.append(row)

        df = pd.DataFrame(processed_rows)
        
        finished = df[df["status"] == "FINISHED"].copy()
        if not finished.empty:
            output_path = self.data_dir / "matches_api_backup.csv"
            finished.to_csv(output_path, index=False)
            logger.info(f"Saved {len(finished)} finished matches to {output_path}")

        scheduled = df[df["status"] == "SCHEDULED"].copy()
        if not scheduled.empty:
            scheduled["hour"] = pd.to_datetime(scheduled["time"], format="%H:%M").dt.hour
            scheduled["venue_code"] = 1
            scheduled["opp_code"] = 0
            scheduled["day_code"] = pd.to_datetime(scheduled["date"]).dt.dayofweek
            
            output_path = self.data_dir / "upcoming_matches.csv"
            scheduled.to_csv(output_path, index=False)
            logger.info(f"Saved {len(scheduled)} upcoming fixtures from API to {output_path}")

    def _get_result(self, score_data):
        """Map API winner field to W/D/L format."""
        winner_map = {"HOME_TEAM": "W", "AWAY_TEAM": "L", "DRAW": "D"}
        return winner_map.get(score_data.get("winner"))

if __name__ == "__main__":
    scraper = FootballAPIScraper()
    scraper.scrape_premier_league()