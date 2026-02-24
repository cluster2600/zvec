# Sprint 1: FAISS GPU Integration - User Stories

## US1: Installation de FAISS GPU

**En tant que** développeur,
**Je veux** installer FAISS GPU facilement via pip,
**Afin que** je puisse immédiatement utiliser l'accélération GPU sans configuration complexe.

### Critères d'acceptation
- [ ] `pip install zvec[gpu]` installe FAISS GPU
- [ ] Détection automatique du GPU NVIDIA
- [ ] Message d'erreur clair si GPU non disponible

### Tasks
- [x] Mettre à jour pyproject.toml
- [x] Ajouter script de vérification GPU
- [x] Créer message d'erreur descriptif

---

## US2: Détection automatique du hardware

**En tant que** développeur,
**Je veux** que zvec détecte automatiquement le meilleur backend disponible,
**Afin que** je n'ai pas à configurer manuellement CPU vs GPU.

### Critères d'acceptation
- [ ] Détection automatique NVIDIA GPU → FAISS GPU
- [ ] Détection AMD GPU → FAISS ROCm (si disponible)
- [ ] Fallback CPU si aucun GPU

### Tasks
- [x] Créer module `zvec.backends`
- [x] Ajouter logging de quel backend est utilisé

---

## US3: Création d'index GPU optimisé

**En tant que** développeur,
**Je veux** créer des indexes optimisés pour GPU,
**Afin d'obtenir les meilleures performances de recherche.

### Critères d'acceptation
- [ ] Support IVF-PQ sur GPU
- [ ] Support HNSW sur GPU (si FAISS supporté)
- [ ] Paramètres configurables (nlist, nprobe, M)

### Tasks
- [ ] Wrapper pour GpuIndexIVF
- [ ] Wrapper pour GpuIndexHNSW
- [ ] Tests de performance

---

## US4: Fallback CPU automatique

**En tant que** développeur,
**Je veux** que zvec bascule automatiquement en CPU si le GPU échoue,
**Afin que** mon application continue à fonctionner sans interruption.

### Critères d'acceptation
- [x] Détection erreur GPU
- [x] Retry automatique sur CPU
- [x] Logging de l'échec GPU

### Tasks
- [x] Implémenter try/except avec fallback
- [x] Ajouter option pour forcer CPU
- [ ] Créer tests de fallback

---

## US5: Benchmarks comparatifs

**En tant que** développeur,
**Je veux** voir des benchmarks comparatifs CPU vs GPU,
**Afin de** mesurer l'amélioration de performance.

### Critères d'acceptation
- [ ] Script de benchmark inclus
- [ ] Résultats pour différentes tailles de datasets
- [ ] Documentation des résultats

### Tasks
- [x] Créer benchmark_runner.py
- [ ] Tester sur 100K, 1M, 10M vecteurs
- [ ] Générer graphiques de comparaison
- [ ] Ajouter à la documentation

---

## Definition of Done Sprint 1

- [ ] Toutes les US complétées
- [ ] Tests unitaires > 90% coverage
- [ ] Tests d'intégration passent
- [ ] Documentation complète
- [ ] Benchmark montre > 5x speedup sur GPU
