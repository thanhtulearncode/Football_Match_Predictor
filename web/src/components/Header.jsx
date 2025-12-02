import { memo } from "react";
import Icon from "./Icon";
import { THEMES } from "../constants";

/**
 * Header Component
 * App header with logo and theme switcher
 */
const Header = ({ currentTheme, onThemeChange, theme }) => {
  return (
    <header
      className={`border-b border-white/5 sticky top-0 z-50 backdrop-blur-md ${theme.bg}/80`}
    >
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg bg-gradient-to-br ${theme.accent}`}>
            <Icon name="activity" className="w-5 h-5 text-slate-900" />
          </div>
          <h1 className="text-xl font-bold tracking-tight">PL Oracle</h1>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:flex gap-1 bg-black/20 p-1 rounded-lg">
            {Object.keys(THEMES).map((themeKey) => (
              <button
                key={themeKey}
                onClick={() => onThemeChange(themeKey)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                  currentTheme === themeKey
                    ? "bg-white/10 text-white shadow-sm"
                    : "text-white/40 hover:text-white/70"
                }`}
                aria-label={`Switch to ${THEMES[themeKey].name} theme`}
              >
                {THEMES[themeKey].name}
              </button>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
};

export default memo(Header);
