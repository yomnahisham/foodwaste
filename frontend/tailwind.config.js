/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: '#4ade80',
                secondary: '#1e293b',
                accent: '#3b82f6',
                bg: '#f8fafc',
                surface: '#ffffff',
                text: '#334155',
                'text-light': '#94a3b8',
            }
        },
    },
    plugins: [],
}
