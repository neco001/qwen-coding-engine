# Qwen-Coding Project Context Template

**Cel:** Standaryzacja `.context/_PROJECT_CONTEXT.md` dla sesji qwen-coding.

**Kiedy używać:**
- Na początku nowego projektu
- Przy starcie sesji qwen-coding (architect/code mode)
- Jako living document - aktualizuj po każdej sesji

---

## Struktura Pliku

```markdown
# Project Context

**Generated:** YYYY-MM-DDTHH:MM
**Updated:** YYYY-MM-DDTHH:MM

**Workspace:** `ścieżka/absolutna/lub/względna`

---

## Directory Structure

```
nazwa_projektu/
├── .context/
│   ├── _PROJECT_CONTEXT.md
│   ├── _DATA_CONTEXT.md
│   └── _SESSION_SUPPLEMENT.md
├── src/
├── tests/
├── .venv/
└── ...
```

---

## Tech Stack

**Runtime:** Python/Node.js/etc.  
**Package Manager:** uv/npm/pip/etc.  
**Database:** DuckDB/PostgreSQL/etc.  
**MCP Servers:**
- `mcp--py-executor` - wykonanie Python
- `mcp--duckdb` - zapytania SQL
- `mcp--qwen-coding` - qwen_coder, qwen_architect, qwen_audit

**Tryby Pracy:**
- `ania` - rozmowa, strategia
- `architect` - planowanie, design
- `code` - implementacja
- `debug` - diagnoza
- `audit` - analiza kodu

---

## PATTERNS - Preferencje Kodowe

### Styl kodu
- **Funkcyjny > OOP** - preferuj kompozycję nad dziedziczeniem
- **Explicit > Implicit** - jasne nazwy, jawne typy
- **Single Responsibility** - jedna funkcja = jedno zadanie

### Dokumentacja
- **Format `.md`** - wszystkie instrukcje w markdown
- **Wysoka gęstość informacyjna** - zero wypełniaczy
- **Precyzja** - brak uproszczeń

### Workflow
- **Chunkowanie** - złożone zadania dziel na części ("część 1/6")
- **Pre-generation checkpoint** - przed generowaniem kodu: potwierdź cel i ograniczenia
- **Post-generation audit** - użyj `qwen_diff_audit_staged_tool` przed commit

---

## ANTIPATTERNS - Czego NIE robić

### Kod
- ❌ **Nie dodawaj importów bez pytania** - sprawdź najpierw, co już jest
- ❌ **Nie generuj boilerplate'u** - pytaj o konkretny zakres
- ❌ **Nie zakładaj struktury projektu** - sprawdź `.context/_PROJECT_CONTEXT.md`
- ❌ **Nie rób refaktoryzacji na własną rękę** - to wymaga zgody

### Proces
- ❌ **Nie działaj w tle bez wiedzy** - każde narzędzie wymaga zatwierdzenia
- ❌ **Nie zgaduj** - jak nie wiesz, to zapytaj
- ❌ **Nie przyspieszaj procesu** - jeśli użytkownik zwalnia, dostosuj tempo

---

## DECISIONS - Decyzje Architektoniczne

### YYYY-MM-DD: Nazwa decyzji
**Kontekst:** Krótki opis sytuacji/problemu

**Decyzja:**
- Punkt 1
- Punkt 2
- Punkt 3

**Zapisane w:** `ścieżka/do/pliku.md`

---

## SESSION_LOG - Historia Sesji

### YYYY-MM-DD (bieżąca)
**Temat:** Krótki opis

**Kluczowe momenty:**
1. Moment 1
2. Moment 2
3. Moment 3

**Decyzje:**
- Decyzja 1
- Decyzja 2

---

## NOTES - Uwagi

### Jak pracować z użytkownikiem
1. Punkt 1
2. Punkt 2
3. Punkt 3

### Narzędzia
- **Interpreter:** `ścieżka/do/python`
- **Package Manager:** `uv/npm`
- **Preferuj:** MCP tools nad shell scripts
```

---

## Instrukcja Użycia dla Architekta

### Przed sesją:
1. **Sprawdź czy `.context/_PROJECT_CONTEXT.md` istnieje**
   - Jeśli nie: stwórz z tego template'u
   - Jeśli tak: przeczytaj, zaktualizuj timestamp

2. **Przeczytaj sekcje:**
   - `PATTERNS` - jakie są preferencje kodowe
   - `ANTIPATTERNS` - czego unikać
   - `DECISIONS` - jakie decyzje już zapadły

3. **Zrób `codebase_search`** jeśli potrzebujesz kontekstu z kodu

### W trakcie sesji:
1. **Pre-generation checkpoint:**
   - "Jaki jest cel tej zmiany?"
   - "Czy są jakieś ograniczenia?"
   - "Czy mam sprawdzić istniejące implementacje?"

2. **Chunkowanie:**
   - Dziel złożone zadania na części
   - Pokazuj postęp ("część 2/6")

3. **Odwołuj się do PATTERNS/ANTIPATTERNS:**
   - "Zgodnie z PATTERNS: funkcyjny > OOP, używam kompozycji"

### Po sesji:
1. **Zaktualizuj `_PROJECT_CONTEXT.md`:**
   - `Updated:` timestamp
   - Dodaj wpis do `DECISIONS`
   - Dodaj wpis do `SESSION_LOG`

2. **Zaktualizuj `_SESSION_SUPPLEMENT.md`:**
   - Punkty kotwiczące
   - Ważne momenty
   - Zasady operacyjne

---

## Przykład Wypełnionego PROJECT_CONTEXT

See: `.context/_PROJECT_CONTEXT.md` w workspace `sialababamak`

---

## Anti-Degradation Checklist

Przed commit:
- [ ] `qwen_diff_audit_staged_tool` uruchomiony
- [ ] Brak regresji w baseline
- [ ] Decyzje zapisane w `DECISIONS`
- [ ] `SESSION_LOG` zaktualizowany
