import { useState, useMemo, useCallback } from "react";
import Header from "./components/Header";
import Tabs from "./components/Tabs";
import ErrorMessage from "./components/ErrorMessage";
import LoadingSpinner from "./components/LoadingSpinner";
import EmptyState from "./components/EmptyState";
import MatchFilters from "./components/MatchFilters";
import PredictionCard from "./components/PredictionCard";
import CustomPrediction from "./components/CustomPrediction";
import { useTheme } from "./hooks/useTheme";
import { useTeams } from "./hooks/useApi";
import { useUpcomingMatches } from "./hooks/useApi";
import { useCustomPrediction } from "./hooks/useApi";
import {
  FILTER_OPTIONS,
  CONFIDENCE_THRESHOLD,
  DEFAULT_MATCH_HOUR,
  DEFAULT_DAY_CODE,
} from "./utils/constants";
import { getMatchKey } from "./utils/matchUtils";

/**
 * Main App Component
 * Orchestrates the entire application with optimized state management
 */
const App = () => {
  const [activeTab, setActiveTab] = useState("upcoming");
  const [filterConfidence, setFilterConfidence] = useState(FILTER_OPTIONS.ALL);

  // Custom hooks for data fetching
  const { teams, loading: teamsLoading, error: teamsError } = useTeams();
  const {
    matches,
    loading: matchesLoading,
    error: matchesError,
  } = useUpcomingMatches();
  const {
    predict: handleCustomPredict,
    result: customResult,
    loading: customLoading,
    error: customError,
    reset: resetCustomPrediction,
  } = useCustomPrediction();

  // Theme management
  const { currentTheme, theme, setTheme } = useTheme();

  // Memoized error handling - combine all errors
  const error = useMemo(() => {
    return teamsError || matchesError || customError || null;
  }, [teamsError, matchesError, customError]);

  // Memoized loading state
  const isLoading = useMemo(() => {
    if (activeTab === "upcoming") {
      return teamsLoading || matchesLoading;
    }
    return teamsLoading || customLoading;
  }, [activeTab, teamsLoading, matchesLoading, customLoading]);

  // Memoized filtered matches
  const filteredMatches = useMemo(() => {
    if (filterConfidence === FILTER_OPTIONS.HIGH_CONFIDENCE) {
      return matches.filter(
        (match) => match.confidence >= CONFIDENCE_THRESHOLD
      );
    }
    return matches;
  }, [matches, filterConfidence]);

  // Handlers with useCallback to prevent unnecessary re-renders
  const handleTabChange = useCallback(
    (tab) => {
      setActiveTab(tab);
      // Reset custom prediction when switching tabs
      if (tab !== "custom") {
        resetCustomPrediction();
      }
    },
    [resetCustomPrediction]
  );

  const handleFilterChange = useCallback((value) => {
    setFilterConfidence(value);
  }, []);

  const handleErrorDismiss = useCallback(() => {
    // Error dismissal can be handled by individual hooks if needed
    // For now, this is a placeholder for future enhancement
  }, []);

  const handleCustomPredictWrapper = useCallback(
    (homeTeam, awayTeam) => {
      handleCustomPredict(
        homeTeam,
        awayTeam,
        DEFAULT_MATCH_HOUR,
        DEFAULT_DAY_CODE
      );
    },
    [handleCustomPredict]
  );

  return (
    <div
      className={`min-h-screen pb-12 transition-colors duration-500 ${theme.bg} ${theme.text}`}
    >
      <Header
        currentTheme={currentTheme}
        onThemeChange={setTheme}
        theme={theme}
      />

      <main className="container mx-auto px-4 mt-8">
        <Tabs
          activeTab={activeTab}
          onTabChange={handleTabChange}
          theme={theme}
        />

        <ErrorMessage message={error} onDismiss={handleErrorDismiss} />

        {activeTab === "upcoming" ? (
          <div className="animate-fade-in" id="panel-upcoming">
            <MatchFilters
              filterValue={filterConfidence}
              onFilterChange={handleFilterChange}
              matchCount={filteredMatches.length}
              theme={theme}
            />

            {isLoading ? (
              <LoadingSpinner theme={theme} />
            ) : (
              // CHANGE: Switched from grid to a centered flex-col list
              <div className="flex flex-col gap-4 max-w-3xl mx-auto">
                {filteredMatches.length > 0 ? (
                  filteredMatches.map((match, index) => (
                    <PredictionCard
                      key={getMatchKey(match, index)}
                      match={match}
                      theme={theme}
                    />
                  ))
                ) : (
                  <EmptyState theme={theme} />
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="animate-fade-in" id="panel-custom">
            <CustomPrediction
              teams={teams}
              onPredict={handleCustomPredictWrapper}
              loading={isLoading}
              result={customResult}
              theme={theme}
            />
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
