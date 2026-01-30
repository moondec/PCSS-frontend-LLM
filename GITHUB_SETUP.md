# Jak skonfigurować i wysłać zmiany na GitHub

Masz już założone konto na GitHub, więc proces jest prosty.

## 1. Utwórz nowe repozytorium na GitHub
1.  Zaloguj się na [github.com](https://github.com).
2.  Kliknij ikonę **+** w prawym górnym rogu i wybierz **New repository**.
3.  Wpisz nazwę (np. `bielik-pcss-app`).
4.  Zaznacz **Private** (jeśli nie chcesz udostępniać kodu publicznie) lub **Public**.
5.  **NIE** zaznaczaj "Initialize with README", "Add .gitignore" itp. (masz już te pliki lokalnie).
6.  Kliknij **Create repository**.

## 2. Podłącz swoje lokalne repozytorium
Skopiuj link do repozytorium (np. `https://github.com/TwojNick/bielik-pcss-app.git`) i wykonaj w terminalu w folderze projektu:

```bash
# Zamień URL na swój!
git remote add origin https://github.com/TwojNick/bielik-pcss-app.git
git branch -M main
git push -u origin main
```

Jeśli dostaniesz błąd "remote origin already exists", wykonaj najpierw:
```bash
git remote remove origin
```
a potem powtórz kroki wyżej.

## 3. Logowanie (jeśli zapyta)
Jeśli terminal zapyta o hasło, a masz włączone 2FA (lub po prostu dla wygody), nie wpisuj hasła do konta. Zamiast tego:
1.  Wejdź w [GitHub Settings -> Developer Settings -> Personal access tokens (Classic)](https://github.com/settings/tokens).
2.  Wygeneruj nowy token ("Generate new token").
3.  Zaznacz uprawnienia `repo`.
4.  Wklej ten token jako hasło w terminalu.

Alternatywnie, jeśli używasz GitHub Desktop lub VS Code, one zajmą się logowaniem za Ciebie.
