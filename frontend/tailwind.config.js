/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Two neutrals and one restrained accent. Nothing else.
        ink: {
          950: "#0A0A0B",
          900: "#101012",
          850: "#161619",
          800: "#1D1D21",
          700: "#2A2A30",
          600: "#3A3A42",
          500: "#6B6B76",
          400: "#8E8E99",
          300: "#B4B4BD",
          100: "#E8E8EC",
          50: "#F6F6F8",
        },
        accent: {
          DEFAULT: "#E8B341",
          soft: "#F0CE83",
          dim: "#8A6A25",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
