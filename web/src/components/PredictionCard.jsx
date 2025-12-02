import { memo, useMemo } from "react";
import Icon from "./Icon";
import ConfidenceBar from "./ConfidenceBar";
import { formatMatchDate, getPredictionConfig } from "../utils/matchUtils";

const PredictionCard = ({ match, theme }) => {
  const formattedDate = useMemo(
    () => formatMatchDate(match.date),
    [match.date]
  );

  const predictionConfig = useMemo(
    () => getPredictionConfig(match.prediction),
    [match.prediction]
  );

  const { badgeColor, borderColor, badgeText } = predictionConfig;

  return (
    <div
      className={`glass-panel ${theme.panel} p-4 rounded-xl border-l-4 ${borderColor} relative group hover:bg-white/5 transition-colors`}
    >
      {/* Header: Date and Time */}
      <div className={`flex items-center gap-2 text-xs ${theme.subtext} mb-3`}>
        <Icon name="calendar" className="w-3 h-3" />
        <span className="font-mono opacity-80">
          {formattedDate} • {match.time}
        </span>
      </div>

      {/* Main Matchup Row */}
      <div className="flex items-center justify-between mb-3">
        {/* Home Team */}
        <div className={`flex-1 text-right font-bold text-lg ${theme.text}`}>
          {match.home_team}
        </div>

        {/* Score/Badge Prediction */}
        <div className="mx-4 flex flex-col items-center">
          <span
            className={`text-[10px] font-bold px-3 py-1 rounded-full text-white ${badgeColor} shadow-md min-w-[80px] text-center`}
          >
            {badgeText}
          </span>
        </div>

        {/* Away Team */}
        <div className={`flex-1 text-left font-bold text-lg ${theme.text}`}>
          {match.away_team}
        </div>
      </div>

      {/* Probability Bar */}
      <div className="opacity-80 hover:opacity-100 transition-opacity">
        <ConfidenceBar probabilities={match.probabilities} theme={theme} />
      </div>
    </div>
  );
};

export default memo(PredictionCard);
