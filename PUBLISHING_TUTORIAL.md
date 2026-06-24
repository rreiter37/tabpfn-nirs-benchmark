# Tutoriel : publier les scripts de reproduction sur GitHub

Objectif : créer un dépôt GitHub **public** contenant les scripts de reproduction
nirs4all (ce dossier), puis insérer son URL dans la section *Data and code
availability* de l'article.

> Pré-requis : un compte GitHub (gratuit) et `git` installé (`git --version`).
> `gh` (GitHub CLI) n'est pas installé sur cette machine : on utilise l'interface
> web + `git` en ligne de commande.

---

## Étape 1 — Préparer un dossier propre et isolé

Le dépôt ne doit contenir **que** les scripts de reproduction (pas tout le monorepo
de thèse). On copie ce dossier vers un emplacement séparé :

```bash
# Copier uniquement les scripts de repro vers un nouveau dossier
cp -r /home/robinr/Desktop/VSCode/CIRAD_PhD_Robin/scripts/Benchmark_tabpfn_nirs4all \
      ~/tabpfn-nirs-benchmark
cd ~/tabpfn-nirs-benchmark

# Nettoyer les artefacts éventuels (cache Python, workspace nirs4all)
rm -rf __pycache__ workspace results_* *.duckdb*
ls   # doit montrer : benchmark_common.py, run_*.py, README.md, requirements.txt, .gitignore
```

## Étape 2 — Ajouter une licence

ACA / Elsevier apprécient une licence explicite. Le code de la thèse et nirs4all
sont sous **CeCILL-2.1** ; le plus simple pour du code de recherche réutilisable est
**MIT** (permissive). Crée un fichier `LICENSE` :

```bash
# Option MIT (remplace l'année et le nom si besoin)
cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2026 Robin Reiter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

> Si tu préfères rester aligné sur nirs4all (CeCILL-2.1), copie plutôt le fichier
> `LICENSE` du clone nirs4all. En cas de doute, MIT est le choix le plus simple et
> le plus accepté pour du code académique.

## Étape 3 — Initialiser le dépôt git local

```bash
cd ~/tabpfn-nirs-benchmark
git init
git add .
git status          # vérifie : pas de workspace/, pas de *.ckpt, pas de __pycache__
git commit -m "Initial commit: nirs4all reproduction of the TabPFN NIR benchmark"
```

## Étape 4 — Créer le dépôt vide sur GitHub (interface web)

1. Va sur <https://github.com/new> (connecté à ton compte).
2. **Repository name** : `tabpfn-nirs-benchmark` (ou le nom que tu veux ; note-le).
3. **Description** : `nirs4all reproduction scripts for the TabPFN NIR calibration benchmark`.
4. Visibilité : coche **Public** (obligatoire pour la *data availability*).
5. **NE coche PAS** « Add a README », « Add .gitignore », « Choose a license »
   (on les a déjà localement, sinon tu auras un conflit au push).
6. Clique **Create repository**.

GitHub affiche alors une page « …or push an existing repository from the command
line » : c'est ce qu'on fait à l'étape suivante.

## Étape 5 — Lier le dépôt local et pousser

Remplace `<TON-COMPTE>` par ton identifiant GitHub réel.

```bash
cd ~/tabpfn-nirs-benchmark
git branch -M main
git remote add origin https://github.com/<TON-COMPTE>/tabpfn-nirs-benchmark.git
git push -u origin main
```

GitHub demandera une authentification :
- **Login web** : un navigateur s'ouvre, tu confirmes — le plus simple.
- **Sinon (token)** : crée un *Personal Access Token* sur
  <https://github.com/settings/tokens> (« Tokens (classic) » → *Generate new token*
  → coche la portée **repo**), et utilise-le comme mot de passe au push.

Vérifie ensuite sur `https://github.com/<TON-COMPTE>/tabpfn-nirs-benchmark` que les
fichiers sont bien là.

## Étape 6 (recommandée) — Archiver une version figée avec un DOI Zenodo

Un dépôt GitHub seul n'est pas *persistant* (il peut être modifié/supprimé). Pour la
version finale ACA, un **DOI** est fortement valorisé. C'est gratuit et rapide :

1. Connecte-toi sur <https://zenodo.org> avec ton compte GitHub.
2. Menu *(ton nom)* → **GitHub** : <https://zenodo.org/account/settings/github/>.
3. Active le bouton **ON** en face de `tabpfn-nirs-benchmark`.
4. Reviens sur GitHub → page du dépôt → **Releases** → **Create a new release** :
   - *Tag* : `v1.0.0`  — *Title* : `Article reproduction scripts v1.0.0`.
   - **Publish release**.
5. Zenodo détecte la release et génère automatiquement un **DOI** (visible sur la
   page Zenodo du dépôt, sous forme d'un badge `10.5281/zenodo.XXXXXXX`).

Tu pourras alors citer **le dépôt GitHub** (toujours à jour) **et** le **DOI Zenodo**
(snapshot figé) dans l'article.

## Étape 7 — Mettre l'URL dans l'article

Dans `manuscript_ACA.tex`, section *Data and code availability*, remplace l'URL
placeholder par l'URL réelle du dépôt :

```latex
... are openly available at \url{https://github.com/<TON-COMPTE>/tabpfn-nirs-benchmark}.
```

Si tu as fait l'étape 6, ajoute la phrase :

```latex
An archived snapshot of the exact version used in this study is available at
\url{https://doi.org/10.5281/zenodo.XXXXXXX}.
```

---

## Mémo : que mettre / ne pas mettre dans le dépôt

| Inclure | Ne pas inclure |
|---|---|
| `benchmark_common.py`, `run_*.py` | Le checkpoint TabPFN `*.ckpt` (trop lourd ; documenter comment l'obtenir) |
| `README.md`, `requirements.txt`, `.gitignore`, `LICENSE` | Le dossier `workspace/` (artefacts DuckDB) |
| (option) un petit dataset d'exemple ou les partitions SPXY | Les gros datasets (ils sont publics : liens en Supplementary) |
| (option) `PUBLISHING_TUTORIAL.md` | `__pycache__/`, `results_*/` |

> Les datasets ne sont **pas** versionnés dans le dépôt : ils sont publics et leurs
> liens d'accès figurent dans le Matériel Supplémentaire de l'article. Le dépôt
> contient le *code* qui les consomme via `--data-root`.
