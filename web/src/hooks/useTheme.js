import { useState, useMemo, useEffect } from "react";
import { THEMES } from "../constants";

/**
 * Custom hook for theme management with localStorage persistence
 * @param {string} defaultTheme - Default theme key
 * @returns {Object} { currentTheme, theme, setTheme, availableThemes }
 */
export const useTheme = (defaultTheme = "slate") => {
  const [currentTheme, setCurrentTheme] = useState(() => {
    // Load from localStorage or use default
    const savedTheme = localStorage.getItem("pl-oracle-theme");
    return savedTheme && THEMES[savedTheme] ? savedTheme : defaultTheme;
  });

  const theme = useMemo(() => THEMES[currentTheme], [currentTheme]);

  // Persist theme to localStorage
  useEffect(() => {
    localStorage.setItem("pl-oracle-theme", currentTheme);
  }, [currentTheme]);

  return {
    currentTheme,
    theme,
    setTheme: setCurrentTheme,
    availableThemes: Object.keys(THEMES),
  };
};
