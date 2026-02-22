# Benchmark Plan: Python 3.14 Features for zvec

## Features à tester

### 1. compression.zstd (PEP 784)
- **Description**: Nouveau module stdlib pour compression Zstandard
- **Use case**: Compression des vecteurs sur disque
- **Avantages**: 
  - Compression très rapide
  - Ratio comparable à gzip
  - Support natif dans stdlib Python 3.14

### 2. base64.z85 (Python 3.13)
- **Description**: Encodage Z85 plus compact que base64
- **Use case**: Stockage de vecteurs binaires
- **Avantages**:
  - 10% plus compact que base64
  - Plus rapide que base64 standard

## Méthodologie Benchmark

### Test 1: compression.zstd
```python
# Comparer:
# - numpy.save (actuel)
# - numpy.save + compression.zstd
# - numpy.save + gzip
# Métriques: taille fichier, temps compression, temps décompression
```

### Test 2: base64.z85
```python
# Comparer:
# - base64.b64encode (actuel)
# - base64.z85encode
# Métriques: taille output, temps encodage, temps décodage
```

## Résultats attendus

| Feature | Amélioration attendue |
|---------|---------------------|
| compression.zstd | 20-30% réduction taille |
| base64.z85 | 10% réduction taille |

## Prochaines étapes

1. Créer benchmark script
2. Exécuter tests
3. Analyser résultats
4. Si amélioration significative → implémenter
5. Créer PR
