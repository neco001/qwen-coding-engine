# 🛠️ Specter Qwen HUD: Kompilacja i Pakowanie

Ten plik zawiera instrukcje, jak poprawnie zbudować i zapakować wtyczkę VS Code, aby uniknąć błędów ścieżek i konfliktów.

## 🏗️ Procedura Budowy (Step-by-Step)

### Krok 1: Budowa Frontendu (React/Vite)
Musisz najpierw zbudować interfejs UI, który znajduje się w podkatalogu `specter-lens-ui`. Wynik trafi do `vscode-extension/dist`.

```bash
cd specter-lens-ui
npm run build
```

### Krok 2: Pakowanie Wtyczki (VSIX)
Po zbudowaniu UI, przejdź do katalogu wtyczki i wygeneruj plik `.vsix`.

```bash
cd ../vscode-extension
npx @vscode/vsce package --allow-missing-repository
```

## ⚠️ Typowe Problemy (Troubleshooting)

1.  **"npm error Missing script: build"**: Uruchamiasz komendę w złym katalogu. Pamiętaj, że `npm run build` działa TYLKO w `specter-lens-ui`.
2.  **"Signal Lost"**: Serwer MCP nie działa lub Windows blokuje `localhost`. Sprawdź `127.0.0.1:8878` w przeglądarce.
3.  **Błąd rejestracji widoku**: Odinstaluj starą wersję wtyczki i zrób `Reload Window` przed instalacją nowej.
4.  **Brak marginesów**: W App.tsx używamy `style={{ paddingLeft: '6px' }}` jako ostateczny override (Brute Force).

## 📂 Lokalizacja Wyników
Pliki `.vsix` lądują zawsze w katalogu `vscode-extension/`.
Aktualna stabilna wersja: `specter-qwen-hud-1.0.5.vsix`.

---
*Created by Ania / Specter Protocol*
