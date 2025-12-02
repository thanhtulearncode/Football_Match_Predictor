import { memo } from "react";
import Icon from "./Icon";

/**
 * ErrorMessage Component
 * Displays error messages with icon and styling
 */
const ErrorMessage = ({ message, onDismiss }) => {
  if (!message) return null;

  return (
    <div className="max-w-2xl mx-auto bg-rose-500/10 border border-rose-500/20 text-rose-200 p-4 rounded-xl mb-8 flex items-center gap-3 animate-fade-in">
      <Icon name="alert-circle" className="w-5 h-5 text-rose-500 flex-shrink-0" />
      <div className="flex-1">
        <p className="font-bold text-sm">Connection Error</p>
        <p className="text-xs opacity-80">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-rose-400 hover:text-rose-300 transition-colors"
          aria-label="Dismiss error"
        >
          <Icon name="x" className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

export default memo(ErrorMessage);
