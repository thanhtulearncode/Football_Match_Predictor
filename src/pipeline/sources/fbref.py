import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup
import time
import random
import logging
import re
from pathlib import Path
from io import StringIO
from config.settings import settings

logger = logging.getLogger(__name__)

class FbrefScraper:
    def __init__(self, data_dir: Path = settings.RAW_DIR):
        self.data_dir = data_dir
        self.temp_dir = self.data_dir / "temp_teams"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session(impersonate="chrome")

    def _get_soup(self, url: str, retries: int = 3):
        for i in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 429:
                    wait = (i + 1) * 60
                    logger.warning(f"Rate limited (429). Sleeping {wait}s...")
                    time.sleep(wait)
                    continue
                
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')
            except Exception as e:
                logger.error(f"Request failed for {url}: {e}")
                time.sleep(random.uniform(5, 10))
        return None

    def scrape_premier_league(self, start_year: int, end_year: int):
        for year in range(start_year, end_year - 1, -1):
            season_str = f"{year}-{year+1}"
            logger.info(f"--- Targeting Season: {season_str} ---")
            
            season_url = f"{settings.FBREF_BASE_URL}/en/comps/9/{season_str}/{season_str}-Premier-League-Stats"
            soup = self._get_soup(season_url)
            
            if not soup:
                continue

            try:
                standings_table = soup.select('table.stats_table')[0]
                links = [l.get("href") for l in standings_table.find_all('a') if '/squads/' in l.get("href", "")]
                team_urls = list(set([f"{settings.FBREF_BASE_URL}{l}" for l in links]))
            except IndexError:
                logger.error(f"Stats table not found for {season_str}")
                continue

            for idx, team_url in enumerate(team_urls):
                team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
                safe_team_name = re.sub(r'[^\w\-_\. ]', '_', team_name)
                
                # Checkpoint
                team_file = self.temp_dir / f"{year}_{safe_team_name}.csv"
                if team_file.exists():
                    logger.info(f"[{idx+1}/{len(team_urls)}] Skipping {team_name} (Checkpoint exists)")
                    continue

                df_team = self._process_team_data(team_url, team_name)
                if df_team is not None and not df_team.empty:
                    df_team["Season"] = year
                    df_team.to_csv(team_file, index=False)
                    logger.info(f"[{idx+1}/{len(team_urls)}] {team_name}: Saved {len(df_team)} matches")
                
                time.sleep(random.uniform(3, 5))

        self._consolidate_data()

    def _process_team_data(self, team_url: str, team_name: str):
        try:
            response = self.session.get(team_url, timeout=30)
            response.raise_for_status()
            matches = pd.read_html(StringIO(response.text), match="Scores & Fixtures")[0]
        except Exception:
            return None

        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [
                l.get("href") for l in soup.find_all('a')
                if l.get("href") and 'all_comps/shooting/' in l.get("href")
            ]
            
            if links:
                shooting_url = f"{settings.FBREF_BASE_URL}{links[0]}"
                resp_shoot = self.session.get(shooting_url, timeout=30)
                if resp_shoot.status_code == 200:
                    shooting = pd.read_html(StringIO(resp_shoot.text), match="Shooting")[0]
                    if isinstance(shooting.columns, pd.MultiIndex):
                        shooting.columns = shooting.columns.droplevel()
                    
                    matches = matches.merge(
                        shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], 
                        on="Date", how="left"
                    )
        except Exception:
            pass

        if "Comp" in matches.columns:
            matches = matches[matches["Comp"] == "Premier League"].copy()
            
        matches["Team"] = team_name
        return matches

    def _consolidate_data(self):
        """Merges temp files into matches.csv (training) and upcoming_matches.csv."""
        logger.info("Consolidating data...")
        all_files = list(self.temp_dir.glob("*.csv"))
        
        if not all_files:
            logger.warning("No data found to consolidate.")
            return

        final_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
        final_df.columns = [c.lower() for c in final_df.columns]
        
        mask_completed = (final_df["result"].notna()) & (final_df["date"] != "Date")
        completed_matches = final_df[mask_completed].copy()
        
        mask_upcoming = (final_df["result"].isna()) & (final_df["date"] != "Date")
        upcoming_matches = final_df[mask_upcoming].copy()

        matches_path = self.data_dir / settings.RAW_MATCHES_FILE
        completed_matches.to_csv(matches_path, index=False)
        logger.info(f"Training Data: {len(completed_matches)} rows -> {matches_path}")
        
        upcoming_path = self.data_dir / settings.UPCOMING_RAW_FILE
        upcoming_matches.to_csv(upcoming_path, index=False)
        logger.info(f"Upcoming Data: {len(upcoming_matches)} rows -> {upcoming_path}")