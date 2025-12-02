import { memo } from "react";
import Icon from "./Icon";

/**
 * Tabs Component
 * Tab navigation for switching between views
 */
const Tabs = ({ activeTab, onTabChange, theme }) => {
  const tabs = [
    { id: "upcoming", label: "Upcoming Fixtures", icon: "calendar-days" },
    { id: "custom", label: "Custom Prediction", icon: "sliders" },
  ];

  return (
    <div className="flex justify-center mb-10">
      <div
        className={`p-1 rounded-xl inline-flex border border-white/5 ${theme.bar_bg}`}
        role="tablist"
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-bold transition-all ${
              activeTab === tab.id
                ? "bg-white/10 text-white shadow-lg"
                : `${theme.subtext} hover:text-white`
            }`}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
          >
            <Icon name={tab.icon} className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
};

export default memo(Tabs);
