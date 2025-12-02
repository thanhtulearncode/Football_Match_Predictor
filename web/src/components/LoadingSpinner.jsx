import { memo } from "react";
import Icon from "./Icon";

/**
 * LoadingSpinner Component
 * Displays loading state with spinner and message
 */
const LoadingSpinner = ({ message = "Crunching numbers...", theme }) => {
  return (
    <div className={`text-center py-24 ${theme.subtext}`}>
      <Icon
        name="loader-2"
        className="w-12 h-12 animate-spin mx-auto mb-4 opacity-50"
      />
      <p className="animate-pulse">{message}</p>
    </div>
  );
};

export default memo(LoadingSpinner);
