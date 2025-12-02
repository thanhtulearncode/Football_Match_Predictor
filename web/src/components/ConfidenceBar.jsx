import { memo, useMemo } from "react";

/**
 * Helper function to calculate width percentage
 * Ensures minimum width of 5% for visibility
 */
const getWidth = (val) => `${Math.max(5, Math.round(val * 100))}%`;

/**
 * Helper function to format percentage
 */
const getPercentage = (val) => `${Math.round(val * 100)}%`;

/**
 * ConfidenceBar Component
 * Displays probability bars for match outcomes (Home Win, Draw, Away Win)
 */
const ConfidenceBar = ({ probabilities, theme }) => {
  // Memoize calculated values to avoid recalculation on every render
  const { homeWidth, drawWidth, awayWidth, homePercent, drawPercent, awayPercent } = useMemo(() => {
    if (!probabilities) {
      return {
        homeWidth: "33%",
        drawWidth: "34%",
        awayWidth: "33%",
        homePercent: "33%",
        drawPercent: "34%",
        awayPercent: "33%",
      };
    }

    return {
      homeWidth: getWidth(probabilities.home_win || 0),
      drawWidth: getWidth(probabilities.draw || 0),
      awayWidth: getWidth(probabilities.away_win || 0),
      homePercent: getPercentage(probabilities.home_win || 0),
      drawPercent: getPercentage(probabilities.draw || 0),
      awayPercent: getPercentage(probabilities.away_win || 0),
    };
  }, [probabilities]);

  return (
    <div className="w-full mt-3">
      <div
        className={`flex h-2 w-full rounded-full overflow-hidden ${theme.bar_bg} mb-2`}
      >
        <div
          className="h-full bg-emerald-500 transition-all duration-500"
          style={{ width: homeWidth }}
          aria-label={`Home win: ${homePercent}`}
        />
        <div
          className="h-full bg-slate-400/50 transition-all duration-500"
          style={{ width: drawWidth }}
          aria-label={`Draw: ${drawPercent}`}
        />
        <div
          className="h-full bg-rose-500 transition-all duration-500"
          style={{ width: awayWidth }}
          aria-label={`Away win: ${awayPercent}`}
        />
      </div>

      <div className="flex justify-between text-xs font-medium">
        <div className="text-emerald-400 flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-emerald-500" />
          Home {homePercent}
        </div>
        <div className={`${theme.subtext} flex items-center gap-1`}>
          <div className="w-2 h-2 rounded-full bg-slate-400" />
          Draw {drawPercent}
        </div>
        <div className="text-rose-400 flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-rose-500" />
          Away {awayPercent}
        </div>
      </div>
    </div>
  );
};

export default memo(ConfidenceBar);
