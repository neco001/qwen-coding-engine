# Storyboard: Protokół Lachmana w akcji (LinkedIn Edition)

Ten dokument zawiera polską wersję symulacji Protokołu Lachmana, zoptymalizowaną pod format karuzeli na LinkedIn. Dodatkowo przygotowałam prompty dla modelu graficznego, aby stworzyć spójną wizualną opowieść.

---

### **Scena 1: Wizja i Strategia**

**Użytkownik:** "Gemini, muszę napisać bezpieczny serwis autoryzacji w FastAPI. Mam dość kopiowania niedziałających kawałków kodu z dokumentacji."

**Gemini:** "Rozumiem. Zróbmy to porządnie. Zamiast improwizować, wezwę **Architekta**, żeby przygotował plan bitwy."

**Gemini Architekt:** `qwen_architect("Build a secure JWT Auth service with FastAPI", context="Standard Python stack")`

**Architekt (Strateg):**
> **Skład powołany:** Senior Security SRE, FastAPI Developer, TDD Specialist.
> **Blueprint:** 
> - **Serce**: Endpoint `/login`, haszowanie haseł BCrypt, generowanie tokenów JWT.
> - **Bezpieczeństwo**: Expiry 15 min, klucze tylko z `.env`.
> - **Clean Slate**: Wykryto stary plik `session_auth.py` – DO USUNIĘCIA, żeby uniknąć konfliktów.
> - **Zasada 80/20**: Odzyskiwanie haseł jest złożone; ląduje w `optional_features` na następny sprint.

> ** Prompt dla Nano-Banana (Image Pro):** 
> *A clean, minimalist dark-mode software chat interface UI. Very realistic screenshot design. Dark gray background (#1e1e1e). Clean sans-serif white text, subtle gray borders, flat design resembling modern developer tools like VS Code or Anthropic Claude. No glowing effects, no neon, no holograms. The image MUST contain exactly this text structured as chat messages:*
> *Użytkownik: "Gemini, muszę napisać bezpieczny serwis autoryzacji w FastAPI. Mam dość kopiowania niedziałających kawałków z dokumentacji."*
> *Gemini: "Rozumiem. Zrobię to porządnie. Wezwę Architekta, żeby przygotował plan bitwy."*
> *Gemini Architekt: `qwen_architect("Build secure JWT Auth service with FastAPI")`*
> *Architekt: "Skład powołany: Security SRE, FastAPI Dev. Blueprint: Endpoint /login, BCrypt, JWT. Security: Expiry 15 min. Clean Slate: Usunąć session_auth.py. 80/20: Odzyskiwanie haseł ląduje w optional_features."*

---

### **Scena 2: Pętla TDD (Faza RED)**

**Użytkownik:** "Plan wygląda solidnie. Wdrażajmy."

**Gemini:** "Czekaj. Zgodnie z Protokołem: **TDD-First**. Najpierw piszemy test, który wybuchnie. To nasz 'dowód prawdy'. Coder, do tablicy."

**Gemini Coder:** `qwen_coder("Write a pytest for JWT token validation based on the Blueprint", context=blueprint)`

**Coder:** 
> Generuje `tests/test_auth.py` z testem sprawdzającym, czy wygasły token jest odrzucany.

**Gemini [Konstruktywny Błąd]:** `pytest tests/test_auth.py`
> **WYNIK:** **FAILED** (ImportError: No module named 'AuthService').
> 
**Gemini:** "Test oblał! To nasza **Faza RED**. Teraz dokładnie wiemy, co musimy zbudować, żeby zapaliło się zielone światło."

> ** Prompt dla Nano-Banana (Image Pro):** 
> *A clean, minimalist dark-mode software chat interface UI. Very realistic screenshot design. Top half shows a chat message. Bottom half shows a dark terminal window with red text output. No neon, no holograms, pure flat UI.*
> *The image MUST contain exactly this text:*
> *Gemini Coder: `qwen_coder("Write a pytest for JWT token validation based on Blueprint")`*
> *Gemini [Konstruktywny Błąd]: `pytest tests/test_auth.py`*
> *[Terminal Red Text]: RESULT: FAILED (ImportError: No module named 'AuthService').*
> *[Chat interface] Gemini: "Test oblał! To nasza Faza RED. Teraz dokładnie wiemy, co wybudować."*

---

### **Scena 3: Implementacja (Faza GREEN)**

**Gemini:** "Dobra, Coder. Skoro mamy test, napisz serwis, który go przejdzie."

**Gemini Coder:** `qwen_coder("Implement AuthService.py with JWT logic to satisfy the failing test", context=test_code)`

**Coder:**
> Dostarcza `auth_service.py` z kompletną logiką (żadnych placeholderów, żadnych leniwych komentarzy).

**Gemini [Uruchomienie]:** `pytest tests/test_auth.py`
> **WYNIK:** 🟢 **PASSED (2 tests passed)**.

> ** Prompt dla Nano-Banana (Image Pro):** 
> *A clean, minimalist dark-mode software chat interface UI. Bottom half shows a dark terminal window with green success text. Pure flat UI, professional developer environment.*
> *The image MUST contain exactly this text:*
> *Gemini Coder: `qwen_coder("Implement AuthService.py with JWT logic to satisfy failing test")`*
> *Gemini [Uruchomienie]: `pytest tests/test_auth.py`*
> *[Terminal Green Text]: RESULT: PASSED (2 tests passed).*

---

### **Scena 4: Audyt SRE i Backlog**

**Gemini:** "Działa, ale czy to jest 'klasa produkcyjna'? Niech **Audytor** (QwQ) to rozszarpie."

**Gemini Audytor:** `qwen_audit(auth_service_code, context=blueprint)`

**Audytor (Analityk QwQ):**
> **Werdykt:** "Kod jest czysty, testy przechodzą. ALE: brakuje **Token Revocation**. Jeśli ktoś ukradnie laptopa użytkownika, nie możemy zdalnie ubić tokena przed jego wygaśnięciem."
> **Rekomendacja:** Dodaj Redis do blacklistowania tokenów.

**Gemini:** "Implementacja jest pewna, ale Audytor znalazł krytyczny punkt na przyszłość. Już dopisuję to do Backlogu."

**Gemini [Automatyzacja]:** `/TBLG_To_backlog("Add Redis-based JWT revocation to the roadmap")`

> ** Prompt dla Nano-Banana (Image Pro):** 
> *A clean, minimalist dark-mode software chat interface UI. Realistic developer tool screenshot.*
> *The image MUST contain exactly this text:*
> *Gemini Audytor: `qwen_audit(auth_service_code, context=blueprint)`*
> *Audytor (QwQ): "Werdykt: Kod jest czysty. ALE: brakuje Token Revocation. Rekomendacja: Dodaj Redis do blacklistowania tokenów."*
> *Gemini [Automatyzacja]: `/TBLG_To_backlog("Add Redis JWT revocation to roadmap")`*

---

### **Scena 5: Podsumowanie (ROI)**

**Gemini:** "W 15 minut przeszliśmy od mglistego pomysłu do **zweryfikowanego, przetestowanego kodu** z gotową strategią długu technicznego."

**Użytkownik:** "I to bez ani jednego `// ... rest of code here`."

**Gemini:** "To jest właśnie **Protokół Lachmana**. Budujemy aplikacje, a nie tylko o nich rozmawiamy."

> ** Prompt dla Nano-Banana (Image Pro):** 
> *A clean, minimalist dark-mode software chat interface UI. Realistic developer tool screenshot. Simple, bold message block.*
> *The image MUST contain exactly this text:*
> *Gemini: "W 15 minut przeszliśmy od mglistego pomysłu do zweryfikowanego, przetestowanego kodu."*
> *Użytkownik: "I to bez ani jednego '// ... rest of code here' w pliku."*
> *Gemini: "To jest właśnie Protokół Lachmana. Budujemy aplikacje, a nie tylko o nich rozmawiamy."*

---

## Dlaczego to działa na LinkedIn?
1. **Dialog**: Czyta się to jak historię, a nie instrukcję obsługi miksera.
2. **Emocje**: Faza RED (porażka jako sukces) to klasyczny "hook" inżynierski.
3. **Konkret**: Pokazujemy, że AI nie tylko pisze kod, ale też planuje i audytuje.
4. **Marka**: "Protokół Lachmana" brzmi jak coś, co chcesz mieć w swoim stacku.
