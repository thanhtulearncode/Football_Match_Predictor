import { memo } from "react";
import Icon from "./Icon";

/**
 * EmptyState Component
 * Displays empty state message when no data is available
 */
const EmptyState = ({ message = "No matches found matching criteria.", theme }) => {
  return (
    <div
      className={`col-span-full text-center py-16 ${theme.panel} rounded-xl border border-dashed border-white/10`}
    >
      <Icon name="ghost" className="w-10 h-10 mx-auto mb-3 opacity-30" />
      <p className={theme.subtext}>{message}</p>
    </div>
  );
};

export default memo(EmptyState);
