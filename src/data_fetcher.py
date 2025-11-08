"""Football match data fetcher"""
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
import logging
from config import RAW_DIR, UPCOMING_MATCHES_FILE, TEAM_NAME_MAPPINGS

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballDataFetcher:
    """Fetch match data from football-data.org API"""
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided - some features will be limited")
        self.headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
    
    def normalize_team_name(self, team_name: str) -> str:
        """Normalize team name using mappings"""
        for canonical_name, variants in TEAM_NAME_MAPPINGS.items():
            if team_name in variants or team_name == canonical_name:
                return canonical_name
        return team_name

    def fetch_upcoming_matches(self, competition_code: str = "PL", days_ahead: int = 30):
        """Fetch upcoming matches for a competition"""
        if not self.api_key:
            logger.error("API key required for upcoming matches")
            return self.create_sample_upcoming_matches()
        
        try:
            url = f"{self.BASE_URL}/competitions/{competition_code}/matches"
            params = {
                "status": "SCHEDULED",
                "dateFrom": datetime.now().strftime("%Y-%m-%d"),
                "dateTo": (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            }
            
            logger.info(f"Fetching upcoming matches for {competition_code}...")
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 429:
                logger.info("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, headers=self.headers, params=params)
            
            response.raise_for_status()
            data = response.json()
            
            matches = []
            for match in data.get("matches", []):
                home_team = self.normalize_team_name(match["homeTeam"]["name"])
                away_team = self.normalize_team_name(match["awayTeam"]["name"])
                
                matches.append({
                    "date": match["utcDate"][:10],
                    "home_team": home_team,
                    "away_team": away_team,
                    "competition": match["competition"]["name"]
                })
            
            df = pd.DataFrame(matches)
            if not df.empty:
                df.to_csv(UPCOMING_MATCHES_FILE, index=False)
                logger.info(f"✅ Saved {len(df)} upcoming matches to {UPCOMING_MATCHES_FILE}")
            else:
                logger.warning("No upcoming matches found")
                df = self.create_sample_upcoming_matches()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {e}")
            return self.create_sample_upcoming_matches()

    def create_sample_upcoming_matches(self) -> pd.DataFrame:
        """Create sample upcoming matches for testing"""
        sample_matches = [
            {
                "date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "home_team": "Manchester United",
                "away_team": "Liverpool", 
                "competition": "Premier League"
            },
            {
                "date": (datetime.now() + timedelta(days=8)).strftime("%Y-%m-%d"),
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "competition": "Premier League"
            },
            {
                "date": (datetime.now() + timedelta(days=9)).strftime("%Y-%m-%d"), 
                "home_team": "Manchester City",
                "away_team": "Tottenham Hotspur",
                "competition": "Premier League"
            },
            {
                "date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
                "home_team": "Newcastle United",
                "away_team": "Brighton & Hove Albion",
                "competition": "Premier League"
            },
            {
                "date": (datetime.now() + timedelta(days=11)).strftime("%Y-%m-%d"),
                "home_team": "West Ham United",
                "away_team": "AFC Bournemouth",
                "competition": "Premier League"
            }
        ]
        
        df = pd.DataFrame(sample_matches)
        df.to_csv(UPCOMING_MATCHES_FILE, index=False)
        logger.info(f"📝 Created sample upcoming matches: {UPCOMING_MATCHES_FILE}")
        return df

    def fetch_historical_matches(self, competition_code: str = "PL", season: int = 2023):
        """Fetch historical matches for training"""
        if not self.api_key:
            logger.error("API key required for historical matches")
            return pd.DataFrame()
        
        try:
            url = f"{self.BASE_URL}/competitions/{competition_code}/matches"
            params = {
                "season": season,
                "status": "FINISHED"
            }
            
            logger.info(f"Fetching historical matches for {competition_code} {season}...")
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 429:
                logger.info("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, headers=self.headers, params=params)
            
            response.raise_for_status()
            data = response.json()
            
            matches = []
            for match in data.get("matches", []):
                if match["status"] != "FINISHED":
                    continue
                    
                home_team = self.normalize_team_name(match["homeTeam"]["name"])
                away_team = self.normalize_team_name(match["awayTeam"]["name"])
                
                # Determine result
                home_goals = match["score"]["fullTime"]["home"] or 0
                away_goals = match["score"]["fullTime"]["away"] or 0
                
                if home_goals > away_goals:
                    result = 2  # Home win
                elif home_goals < away_goals:
                    result = 0  # Away win
                else:
                    result = 1  # Draw
                
                matches.append({
                    "date": match["utcDate"][:10],
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                    "competition": match["competition"]["name"]
                })
            
            df = pd.DataFrame(matches)
            if not df.empty:
                output_path = RAW_DIR / "matches.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"✅ Saved {len(df)} historical matches to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical matches: {e}")
            return pd.DataFrame()

def main():
    """Main function to fetch data"""
    fetcher = FootballDataFetcher()
    
    print("🏈 Football Data Fetcher")
    print("=" * 50)
    
    # Fetch upcoming matches
    print("📅 Fetching upcoming matches...")
    upcoming_matches = fetcher.fetch_upcoming_matches()
    print(f"✅ Upcoming matches: {len(upcoming_matches)}")
    
    # Fetch historical matches if API key available
    if fetcher.api_key:
        print("📊 Fetching historical matches...")
        historical_matches = fetcher.fetch_historical_matches()
        print(f"✅ Historical matches: {len(historical_matches)}")
    else:
        print("ℹ️  No API key - using existing historical data")
    
    print("🎉 Data fetching complete!")

if __name__ == "__main__":
    main()