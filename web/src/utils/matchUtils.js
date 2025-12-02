/**
 * Match utility functions
 */

/**
 * Formats date to readable string
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted date string
 */
export const formatMatchDate = (date) => {
  const dateObj = typeof date === "string" ? new Date(date) : date;
  return dateObj.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
};

/**
 * Gets prediction badge configuration
 * @param {string} prediction - Prediction type (Home Win, Away Win, Draw)
 * @returns {Object} { badgeColor, borderColor, badgeText }
 */
export const getPredictionConfig = (prediction) => {
  const configs = {
    "Home Win": {
      badgeColor: "bg-emerald-600",
      borderColor: "border-l-emerald-500",
      badgeText: "HOME WIN",
    },
    "Away Win": {
      badgeColor: "bg-rose-600",
      borderColor: "border-l-rose-500",
      badgeText: "AWAY WIN",
    },
    Draw: {
      badgeColor: "bg-slate-500",
      borderColor: "border-l-slate-400",
      badgeText: "DRAW",
    },
  };

  return (
    configs[prediction] || {
      badgeColor: "bg-slate-600",
      borderColor: "border-l-transparent",
      badgeText: "UNKNOWN",
    }
  );
};

/**
 * Checks if prediction has high confidence
 * @param {number} confidence - Confidence value (0-1)
 * @param {number} threshold - Confidence threshold (default: 0.75)
 * @returns {boolean}
 */
export const isHighConfidence = (confidence, threshold = 0.75) => {
  return confidence > threshold;
};

/**
 * Generates unique key for match
 * @param {Object} match - Match object
 * @param {number} index - Fallback index
 * @returns {string} Unique key
 */
export const getMatchKey = (match, index) => {
  if (match.id) return match.id.toString();
  if (match.date && match.home_team && match.away_team) {
    return `${match.date}-${match.home_team}-${match.away_team}`;
  }
  return `match-${index}`;
};
