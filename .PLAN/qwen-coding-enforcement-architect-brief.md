# Qwen-Coding Enforcement: Architect Brief

## Cel Sesji

Dodać **enforcement po stronie serwera MCP** do narzędzi `qwen_architect` i `qwen_coder` w projekcie `qwen-coding-local`.

**Problem:** Agent "biega jak kura" - ignoruje protokół, nie dodaje tasków, koduje bez testów.

**Rozwiązanie:** Wbudować walidację w same narzędzia MCP, nie tylko w tryby Roo Code.

---

## Kontekst

### Obecny stan (`.roo/modes/`):

1. **Architect Mode** - zabrania kodowania, nakazuje dodawanie tasków
2. **Code Mode** - zabrania planowania, nakazuje TDD
3. **Quick-Fix Mode** - pozwala na zmiany ≤10 linii bez protokołu

**Problem:** Tryby są TYLKO po stronie Roo Code. Agent może je zignorować i użyć narzędzi bezpośrednio.

### Plik do modyfikacji:

- `src/qwen_mcp/tools.py` - główna implementacja narzędzi

---

## Wymagane Zmiany

### 1. `qwen_architect` - Auto-Add Tasks

**Obecnie:** Tworzy plan, ale NIE dodaje tasków automatycznie.

**Wymaganie:** Dodaj parametr `auto_add_tasks: bool = True`

```python
async def qwen_architect(
    goal: str,
    context: Optional[str] = None,
    auto_add_tasks: bool = True,  # NOWY PARAMETR
    session_id: Optional[str] = None,
    ctx: Optional[Context] = None
) -> str:
    # ... istniejąca logika ...
    
    # PO stworzeniu planu:
    if auto_add_tasks and plan.tasks:
        sync_engine = DecisionLogSyncEngine(workspace_root=workspace_root, session_id=session_id)
        for task in plan.tasks:
            await sync_engine.add_task(
                task_name=task.name,
                advice=task.description,
                complexity=task.complexity,
                tags=task.tags,
                risk_score=task.risk_score
            )
    
    return plan.to_markdown()
```

**Efekt:** Architect AUTOMATYCZNIE dodaje taski do BACKLOG.md i decision_log.parquet.

---

### 2. `qwen_coder` - Require Plan

**Obecnie:** Koduje bez sprawdzania czy istnieje plan/task.

**Wymaganie:** Dodaj parametr `require_plan: bool = True`

```python
async def qwen_coder(
    prompt: str,
    mode: str = "auto",
    context: Optional[str] = None,
    require_plan: bool = True,  # NOWY PARAMETR
    session_id: Optional[str] = None,
    ctx: Optional[Context] = None
) -> str:
    workspace_root = get_workspace_root(ctx) if ctx else "."
    
    # PRE-FLIGHT CHECK: Czy istnieje pending task?
    if require_plan:
        tasks_df = read_decision_log_parquet(workspace_root)
        pending_tasks = tasks_df[tasks_df['status'] == 'pending'] if 'status' in tasks_df.columns else []
        
        if len(pending_tasks) == 0:
            raise ValueError(
                "No pending tasks found. Run qwen_architect first or set require_plan=False for ad-hoc coding."
            )
    
    # ... istniejąca logika generowania kodu ...
```

**Efekt:** Coder blokuje kodowanie jeśli nie ma pending tasków.

---

### 3. `qwen_coder` - Require Test (TDD Enforcement)

**Obecnie:** Koduje bez sprawdzania czy istnieje test.

**Wymaganie:** Dodaj parametr `require_test: bool = False` (domyślnie False dla backward compatibility)

```python
async def qwen_coder(
    prompt: str,
    mode: str = "auto",
    context: Optional[str] = None,
    require_plan: bool = True,
    require_test: bool = False,  # NOWY PARAMETR
    session_id: Optional[str] = None,
    ctx: Optional[Context] = None
) -> str:
    workspace_root = get_workspace_root(ctx) if ctx else "."
    
    # PRE-FLIGHT CHECK: Czy istnieje test?
    if require_test:
        test_files = list(Path(workspace_root).glob("test_*.py")) + \
                     list(Path(workspace_root).glob("*_test.py"))
        
        if len(test_files) == 0:
            raise ValueError(
                "No test files found. Write a test first (TDD RED phase) or set require_test=False."
            )
    
    # ... istniejąca logika ...
```

**Efekt:** Coder wymusza TDD - najpierw test, potem kod.

---

### 4. Pre-Flight Check dla Wszystkich Narzędzi

**Wymaganie:** Każde narzędzie powinno wołać `qwen_init_request` na początku.

```python
async def qwen_architect(...):
    # INIT REQUEST - mandatory
    if ctx:
        await qwen_init_request(ctx=ctx)
    
    # ... reszta logiki ...

async def qwen_coder(...):
    # INIT REQUEST - mandatory
    if ctx:
        await qwen_init_request(ctx=ctx)
    
    # ... reszta logiki ...
```

**Efekt:** HUD telemetry działa poprawnie, token counters są resetowane.

---

## Architektura Zmian

### Warstwy Enforcementu:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Roo Code Modes                                 │
│ - 🏗️-architect.md: blokuje narzędzia kodujące          │
│ - 💻-code.md: blokuje narzędzia planujące               │
│ - 🔧-quick-fix.md: ogranicza zakres zmian               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: MCP Tool Validation (TA SESJA)                 │
│ - qwen_architect: auto_add_tasks=True                   │
│ - qwen_coder: require_plan=True, require_test=True      │
│ - Wszystkie: init_request na początku                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Anti-Degradation (istniejący)                  │
│ - qwen_diff_audit_tool: wykrywa regresje PO fakcie      │
│ - qwen_create_baseline_tool: snapshot przed zmianami    │
└─────────────────────────────────────────────────────────┘
```

---

## Kryteria Akceptacji

### Dla `qwen_architect`:
- [ ] Parametr `auto_add_tasks` domyślnie `True`
- [ ] Taski są dodawane do BACKLOG.md i decision_log.parquet
- [ ] Jeśli `auto_add_tasks=False`, zachowuje się jak obecnie

### Dla `qwen_coder`:
- [ ] Parametr `require_plan` domyślnie `True`
- [ ] Blokuje jeśli brak pending tasków
- [ ] Parametr `require_test` domyślnie `False` (backward compatibility)
- [ ] Blokuje jeśli brak testów i `require_test=True`
- [ ] `qwen_init_request` wywoływane na początku

### Dla wszystkich narzędzi:
- [ ] Każde narzędzie woła `qwen_init_request` na początku
- [ ] Błędy pre-flight są jasne i wskazują rozwiązanie

---

## Ryzyka

1. **Breaking Change:** Istniejące skrypty mogą polegać na braku walidacji
   - **Mitigacja:** Parametry domyślnie `True` ale z możliwością wyłączenia

2. **False Positives:** Testy mogą istnieć w innej lokalizacji
   - **Mitigacja:** Rozszerzyć search pattern o `tests/`, `spec/`, etc.

3. **Performance:** Dodatkowe checki mogą spowolnić narzędzia
   - **Mitigacja:** Checki są lekkie (read parquet, glob files)

---

## Następne Kroki (po tej sesji)

1. **Przetestować zmiany:**
   ```bash
   cd C:\Repos\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local
   pytest tests/test_tools_enforcement.py
   ```

2. **Zaktualizować dokumentację:**
   - `docs/TDD.md` - dodać sekcję o enforcement
   - `docs/ARCHITECTURE.md` - zaktualizować diagram warstw

3. **Powiadomić użytkownika:**
   - Tryby Roo Code + Enforcement MCP = pełna ochrona

---

## Instrukcja dla Architekta

**Tryb:** 🏗️ Architect

**Zadanie:**
1. Przeanalizuj `src/qwen_mcp/tools.py`
2. Zaprojektuj implementację enforcementu zgodnie z tym briefem
3. Dodaj taski do BACKLOG.md dla każdej zmiany
4. Stwórz baseline snapshot przed modyfikacjami
5. **NIE KODUJ** - przekaż do Code Mode

**Output:**
- Plan implementacji
- Taski w BACKLOG.md
- Baseline snapshot

---

## Meta-Komentarz

To jest **druga połowa walki** z sesji gdzie agent "biegał jak kura".

**Sesja 1 (TERAZ zakończona):**
- ✅ Stworzyła 3 tryby Roo Code
- ✅ Zdiagnozowała brak TDD enforcement w qwen_coder
- ✅ Napisała ten brief

**Sesja 2 (NASTĘPNA):**
- [ ] Zmodyfikuje qwen-coding/tools.py
- [ ] Doda enforcement po stronie MCP
- [ ] Połączy warstwy ochrony

**Cel końcowy:** Agent NIE MOŻE zignorować protokołu - ani przez tryby, ani przez narzędzia.
