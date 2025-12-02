import { memo } from "react";
import { FILTER_OPTIONS } from "../utils/constants";

/**
 * MatchFilters Component
 * Filter controls for match list
 */
const MatchFilters = ({ filterValue, onFilterChange, matchCount, theme }) => {
  return (
    <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-4">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        Upcoming Matches
        <span
          className={`text-sm font-normal px-2 py-0.5 rounded-full bg-white/5 ${theme.subtext}`}
        >
          {matchCount}
        </span>
      </h2>
      <select
        className={`${theme.bg} border border-white/10 rounded-lg px-4 py-2 text-sm ${theme.text} outline-none focus:border-blue-500 transition-colors cursor-pointer`}
        value={filterValue}
        onChange={(e) => onFilterChange(e.target.value)}
        aria-label="Filter matches by confidence level"
      >
        <option value={FILTER_OPTIONS.ALL}>All Confidence Levels</option>
        <option value={FILTER_OPTIONS.HIGH_CONFIDENCE}>
          🔥 High Confidence Only
        </option>
      </select>
    </div>
  );
};

export default memo(MatchFilters);
