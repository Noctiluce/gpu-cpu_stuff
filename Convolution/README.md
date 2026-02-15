# Programmation CUDA pour GPU

Ce README présente les concepts essentiels de la programmation CUDA, la hiérarchie mémoire, le rôle des Streaming Multiprocessors (SM), les relations entre threads, blocks et warps, et les bonnes pratiques pour optimiser la performance.

---

## 1. Concepts de base

### 1.1 Threads
- Un **thread** est l’unité d’exécution de base sur le GPU.
- Chaque thread exécute une instance du kernel.
- Identifiant unique dans un block : `threadIdx`.

### 1.2 Blocks
- Un **block** est un regroupement de threads.
- Chaque thread dans un block est identifié par `threadIdx`.
- Taille du block (`blockDim`) fixe au lancement du kernel.
- Maximum de threads par block dépend de l’architecture (ex. 1024 threads pour les architectures modernes).

### 1.3 Warps
- Un **warp** est un groupe de 32 threads exécutés simultanément sur un SM.
- Divergence dans un warp (ex. `if` différents) réduit la performance.

### 1.4 Grids
- Une **grid** est un ensemble de blocks.
- Chaque block dans la grid est identifié par `blockIdx`.
- Taille de la grid (`gridDim`) définit le nombre total de blocks.

---

## 2. Hiérarchie de mémoire

| Mémoire             | Accès         | Scope           | Latence        | Usage typique |
|--------------------|---------------|----------------|----------------|---------------|
| Registres           | Très rapide   | Par thread      | ~1 cycle       | Variables locales |
| Shared memory       | Rapide        | Par block       | 10-100 cycles  | Communication entre threads d’un block |
| Global memory       | Lent          | Par grid        | 400-600 cycles | Données principales accessibles par tous les threads |
| Constant memory     | Rapide        | Par grid (lecture) | 1 cycle si cache | Paramètres constants |
| Texture memory      | Optimisé pour lecture 2D/3D | Par grid | 1 cycle si cache | Images, textures |

---

## 3. Streaming Multiprocessors (SM)

- Chaque SM contient plusieurs **CUDA cores**.
- Un SM exécute plusieurs warps simultanément.
- Les threads d’un block sont dispatchés sur les SM disponibles.
- Nombre de threads exécutés simultanément dépend de :
  - Taille du block
  - Ressources par thread (registres, shared memory)
  - Capacité du SM

**Trouver le nombre de SM et limites :**
```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
int numSM = prop.multiProcessorCount;
int maxThreadsPerBlock = prop.maxThreadsPerBlock;
int warpSize = prop.warpSize;
```
# CUDA : Threads, Blocks, Warps et SM

## Diagramme simplifié
Grid
 ├── Block 0 ──> SM0 ──> Warps
 ├── Block 1 ──> SM1 ──> Warps
 └── Block 2 ──> SM0 ──> Warps



 ---

## 4. Relations block size, grid size, warps et SM

- **Block size** → nombre de threads par block.
- **Warp** → chaque 32 threads = 1 warp.
- **Grid size** → nombre total de blocks.
- **SM** → reçoit les blocks selon sa capacité et ressources disponibles.

**Exemple :**

- Grid = 128 blocks, Block = 256 threads → 128 × 256 = 32 768 threads.
- SM peut exécuter 2048 threads simultanément → blocks dispatchés par SM selon ressources.

---

## 5. Performance et organisation des threads

### 5.1 Taille des blocks
- Trop petit → sous-utilisation du GPU.
- Trop grand → dépasse les ressources SM (registres, shared memory).

### 5.2 Coalescing mémoire
- Accès global memory aligné par warp → optimal.
- Accès désordonné → latence élevée.

### 5.3 Divergence de branchement
- Threads d’un warp doivent suivre le même chemin pour performance optimale.
- Divergence → exécution séquentielle au lieu de simultanée.

---

## 6. Exemples pratiques

### 6.1 Addition vecteur
```cpp
__global__ void vecAdd(float *A, float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
```

### 6.2 Coalescing

- Chaque thread accède à `A[idx]` de manière contiguë pour chaque warp → accès mémoire efficace.

---

## 7. Bonnes pratiques

- **Maximiser l’occupation des SM** : ajuster la taille des blocks pour utiliser pleinement registres et shared memory.
- **Éviter la divergence dans les warps** pour maintenir l’exécution parallèle.
- **Exploiter le coalescing** des accès à la global memory pour réduire la latence.
- **Préférer la shared memory** pour limiter les accès à la mémoire globale.
- **Vérifier les propriétés du device** pour adapter block/grid size :

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("SM: %d, max threads per block: %d, warp size: %d\n",
       prop.multiProcessorCount, prop.maxThreadsPerBlock, prop.warpSize);
```

# Optimisation GPU avec CUDA : résumé

## Paramètres du GPU étudié
- warpSize = 32 → exécution des threads par groupes de 32
- numSM = 40 → nombre de multiprocesseurs sur le GPU
- maxThreadsPerBlock = 1024 → nombre maximal de threads par bloc
- maxThreadsPerMultiProcessor = 1024 → nombre maximal de threads actifs par SM

---

## 1. Taille de bloc et grille

### Bloc
- Exemple : 16×16 = 256 threads par bloc
- Doit être **multiple de warpSize** (32) pour éviter des threads inactifs
- Chaque bloc = 8 warps (256 ÷ 32)

### Grille
- Exemple : 32×32 = 1024 blocs
- Chaque SM peut gérer plusieurs blocs, mais **limitée à 1024 threads actifs par SM**
- Pour 256 threads par bloc → max 4 blocs actifs par SM simultanément
- Les blocs restants sont mis en queue et exécutés dès qu’un bloc se libère

---

## 2. Répartition sur les SM
- Total blocs : 1024
- SM = 40
- Théorique : 1024 ÷ 40 ≈ 25 blocs par SM
- En pratique : chaque SM ne peut exécuter que 4 blocs à la fois (256×4=1024 threads max)
- Scheduler GPU s’occupe d’exécuter les blocs restants quand des blocs se terminent

---

## 3. Latence mémoire
- Accès mémoire globale = 400–600 cycles → thread en attente
- **Masquer la latence** : avoir plusieurs blocs/warps prêts à calculer pendant que d’autres attendent
- **Over-subscription** : plus de blocs que ce que le SM peut exécuter simultanément pour toujours garder des warps actifs
- Optimisations :
    - Taille de bloc raisonnable (ex : 16×16)
    - Accès mémoire coalescé
    - Utilisation de la mémoire partagée pour réduire les temps d’attente
    - Avoir plus de blocs que le nombre de threads max par SM

---

## 4. Bonnes pratiques
- Taille de bloc : multiple de warpSize, 256–1024 threads selon le GPU
- Nombre de blocs : suffisant pour saturer SM et masquer latence
- Penser à la mémoire partagée pour réduire les accès lents à la mémoire globale
- Profiler et tester différentes tailles de blocs/grilles pour trouver le meilleur compromis

---

## 5. Conclusion
- Bloc 16×16, grille 32×32 → fonctionne bien
- Chaque SM n’exécute que 4 blocs à la fois, les autres blocs attendent
- L’overlap entre calcul et attente mémoire est crucial pour un GPU performant
- Optimiser la répartition des threads et blocs permet de **maximiser l’occupation et la performance**
