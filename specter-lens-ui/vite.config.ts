import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
    plugins: [tailwindcss(), react()],
    build: {
        outDir: '../vscode-extension/dist',
        emptyOutDir: true,
    }
})
