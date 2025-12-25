/**
 * API Service Module
 * Centralized API communication logic
 */

const API_PORT = import.meta.env.VITE_API_PORT || 8000;
const API_BASE = `http://${window.location.hostname}:${API_PORT}`;

/**
 * Fetches list of available teams
 * @returns {Promise<string[]>} Array of team names
 */
export const fetchTeams = async () => {
  try {
    const response = await fetch(`${API_BASE}/teams`);
    if (!response.ok) {
      throw new Error(`Failed to fetch teams: ${response.statusText}`);
    }
    const data = await response.json();
    return data.teams;
  } catch (error) {
    console.error("API Connection Error:", error);
    throw error;
  }
};

/**
 * Fetches upcoming match predictions
 * @returns {Promise<Array>} Array of match predictions
 */
export const fetchUpcomingMatches = async () => {
  try {
    const response = await fetch(`${API_BASE}/predict/upcoming`);
    if (!response.ok) {
      throw new Error(`Failed to fetch matches: ${response.statusText}`);
    }
    const data = await response.json();
    return data.predictions;
  } catch (error) {
    console.error("API Connection Error:", error);
    throw error;
  }
};

/**
 * Predicts match result for custom team matchup
 * @param {string} homeTeam - Home team name
 * @param {string} awayTeam - Away team name
 * @param {number} hour - Match hour (0-23)
 * @param {number} dayCode - Day of week code (0-6)
 * @returns {Promise<Object>} Prediction result
 */
export const predictCustomMatch = async (
  homeTeam,
  awayTeam,
  hour = 15,
  dayCode = 5
) => {
  const response = await fetch(`${API_BASE}/predict/teams`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      home_team: homeTeam,
      away_team: awayTeam,
      hour,
      day_code: dayCode,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Prediction failed: ${response.statusText}`
    );
  }

  return response.json();
};

export default {
  fetchTeams,
  fetchUpcomingMatches,
  predictCustomMatch,
};
