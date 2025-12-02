import { memo, useState, useCallback, useMemo } from "react";
import Icon from "./Icon";
import PredictionCard from "./PredictionCard";

/**
 * CustomPrediction Component
 * Form for custom match predictions with team selection
 */
const CustomPrediction = ({ teams, onPredict, loading, result, theme }) => {
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");

  const isFormValid = useMemo(
    () => homeTeam && awayTeam && homeTeam !== awayTeam,
    [homeTeam, awayTeam]
  );

  const handleSubmit = useCallback(
    (e) => {
      e.preventDefault();
      if (isFormValid) {
        onPredict(homeTeam, awayTeam);
      }
    },
    [homeTeam, awayTeam, isFormValid, onPredict]
  );

  // Memoize available teams for each dropdown
  const availableHomeTeams = useMemo(
    () => teams.filter((team) => team !== awayTeam),
    [teams, awayTeam]
  );

  const availableAwayTeams = useMemo(
    () => teams.filter((team) => team !== homeTeam),
    [teams, homeTeam]
  );

  // Create match object for result display
  const resultMatch = useMemo(
    () =>
      result
        ? {
            ...result,
            date: new Date().toISOString(),
            time: "Custom",
            venue: "Neutral/Sim",
          }
        : null,
    [result]
  );

  return (
    <div className="max-w-2xl mx-auto">
      <div
        className={`glass-panel ${theme.panel} p-8 rounded-2xl mb-8 border border-white/5`}
      >
        <h2
          className={`text-xl font-bold mb-6 flex items-center gap-2 ${theme.text}`}
        >
          <Icon name="swords" className="w-5 h-5 text-blue-400" />
          Custom Matchup
        </h2>

        <form onSubmit={handleSubmit} className="space-y-8" noValidate>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 relative">
            <div>
              <label
                htmlFor="home-team"
                className={`block text-xs font-bold uppercase tracking-wider ${theme.subtext} mb-2`}
              >
                Home Team
              </label>
              <select
                id="home-team"
                className={`w-full ${theme.bg} border border-white/10 rounded-lg p-3 ${theme.text} focus:ring-2 focus:ring-blue-500 outline-none transition-colors`}
                value={homeTeam}
                onChange={(e) => setHomeTeam(e.target.value)}
                aria-required="true"
              >
                <option value="">Select Team</option>
                {availableHomeTeams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </div>

            <div className="hidden md:flex absolute inset-0 items-center justify-center pointer-events-none">
              <div
                className={`${theme.bar_bg} rounded-full p-2 mt-6 border border-white/10`}
              >
                <span className={`text-xs font-bold ${theme.subtext}`}>VS</span>
              </div>
            </div>

            <div>
              <label
                htmlFor="away-team"
                className={`block text-xs font-bold uppercase tracking-wider ${theme.subtext} mb-2`}
              >
                Away Team
              </label>
              <select
                id="away-team"
                className={`w-full ${theme.bg} border border-white/10 rounded-lg p-3 ${theme.text} focus:ring-2 focus:ring-blue-500 outline-none transition-colors`}
                value={awayTeam}
                onChange={(e) => setAwayTeam(e.target.value)}
                aria-required="true"
              >
                <option value="">Select Team</option>
                {availableAwayTeams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !isFormValid}
            className={`w-full bg-gradient-to-r ${theme.accent} hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 font-bold py-4 rounded-xl transition-all shadow-lg transform active:scale-[0.99]`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <Icon name="loader-2" className="animate-spin w-4 h-4" />
                Analyzing Stats...
              </span>
            ) : (
              "Predict Result"
            )}
          </button>
        </form>
      </div>

      {resultMatch && (
        <div className="animate-fade-in-up">
          <h3
            className={`text-sm font-bold uppercase tracking-widest mb-4 text-center ${theme.subtext}`}
          >
            Prediction Result
          </h3>
          <PredictionCard match={resultMatch} theme={theme} />
        </div>
      )}
    </div>
  );
};

export default memo(CustomPrediction);
