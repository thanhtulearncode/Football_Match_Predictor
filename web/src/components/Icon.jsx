import { memo, useMemo } from "react";
import { icons } from "lucide-react";

/**
 * Converts kebab-case to PascalCase
 * Example: "calendar-days" -> "CalendarDays"
 */
const toPascalCase = (str) =>
  str
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join("");

/**
 * Icon Component
 * Wrapper for Lucide React icons with kebab-case to PascalCase conversion
 */
const Icon = ({ name, className = "w-5 h-5" }) => {
  // Memoize icon name conversion
  const iconName = useMemo(() => toPascalCase(name), [name]);

  // Memoize icon lookup
  const LucideIcon = useMemo(() => icons[iconName], [iconName]);

  if (!LucideIcon) {
    if (process.env.NODE_ENV === "development") {
      console.warn(`Icon "${name}" (${iconName}) not found in lucide-react`);
    }
    return null;
  }

  return <LucideIcon className={className} aria-hidden="true" />;
};

export default memo(Icon);
