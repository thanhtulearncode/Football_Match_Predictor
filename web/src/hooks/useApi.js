import { useState, useEffect, useCallback } from "react";
import * as apiService from "../services/api";

/**
 * Custom hook for fetching teams
 * @returns {Object} { teams, loading, error, refetch }
 */
export const useTeams = () => {
  const [teams, setTeams] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchTeams = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.fetchTeams();
      setTeams(data);
    } catch (err) {
      setError(err.message || "Failed to fetch teams");
      console.error("Error fetching teams:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTeams();
  }, [fetchTeams]);

  return { teams, loading, error, refetch: fetchTeams };
};

/**
 * Custom hook for fetching upcoming matches
 * @returns {Object} { matches, loading, error, refetch }
 */
export const useUpcomingMatches = () => {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchMatches = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.fetchUpcomingMatches();
      setMatches(data);
    } catch (err) {
      setError(err.message || "Failed to fetch matches");
      console.error("Error fetching matches:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMatches();
  }, [fetchMatches]);

  return { matches, loading, error, refetch: fetchMatches };
};

/**
 * Custom hook for custom match prediction
 * @returns {Object} { predict, result, loading, error }
 */
export const useCustomPrediction = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = useCallback(
    async (homeTeam, awayTeam, hour, dayCode) => {
      if (!homeTeam || !awayTeam || homeTeam === awayTeam) {
        setError("Please select two different teams");
        return;
      }

      setLoading(true);
      setError(null);
      setResult(null);

      try {
        const prediction = await apiService.predictCustomMatch(
          homeTeam,
          awayTeam,
          hour,
          dayCode
        );
        setResult(prediction);
      } catch (err) {
        setError(err.message || "Prediction failed");
        console.error("Error predicting match:", err);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { predict, result, loading, error, reset };
};
