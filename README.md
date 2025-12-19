# Fine-Tuning de LLM avec PEFT / LoRA  
**Du Flan-T5 à Llama 3.2-3B (QLoRA)**

## 1. Objectif du projet
L’objectif de ce projet est d’explorer et de comparer différentes stratégies de **fine-tuning de grands modèles de langage (LLM)** en utilisant des techniques **parameter-efficient** (PEFT), et en particulier **LoRA** et **QLoRA**.

Le projet commence par un modèle pédagogique (**Flan-T5-base**) pour comprendre les mécanismes, puis évolue vers un modèle moderne et performant (**Llama 3.2-3B Instruct**) chargé en **4-bits** afin de rester compatible avec des ressources matérielles limitées (Google Colab).

---

## 2. Modèles utilisés

### Modèle 1 – Flan-T5-base
- Architecture : **Encoder–Decoder (Seq2Seq)**
- Avantages : simplicité, rapidité, idéal pour l’apprentissage
- Limites : performances dépassées par les modèles récents

### Modèle 2 – Llama 3.2-3B Instruct
- Architecture : **Decoder-only (CausalLM)**
- Chargement quantifié **4-bits (bitsandbytes / QLoRA)**
- Entraînement via **LoRA**
- Avantages :
  - performances supérieures,
  - faible empreinte mémoire,
  - fine-tuning possible sur un GPU standard

---

## 3. Dataset

- **DialogSum** (`knkarthick/dialogsum`)
- Tâche : **résumé automatique de dialogues**
- Splits :
  - train
  - validation
  - test
- Pour rester dans des temps acceptables :
  - **100 exemples** pour l’entraînement
  - **50 exemples** pour validation / test

---

## 4. Méthodologie

### 4.1 Fine-tuning classique (Flan-T5)
- Fine-tuning complet puis via **PEFT / LoRA**
- Comparaison :
  - modèle original
  - modèle fine-tuné
- Évaluation via **ROUGE**

### 4.2 Fine-tuning moderne (Llama 3.2-3B)
- Quantification **4-bits (QLoRA)** avec `bitsandbytes`
- Gel du modèle de base avec `prepare_model_for_kbit_training`
- Ajout d’adapteurs **LoRA** sur les projections d’attention :
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Entraînement de **~1–2 % des paramètres seulement**

---

## 5. Entraînement

- Framework : **Hugging Face Transformers**
- Optimisation :
  - `paged_adamw_8bit`
  - `gradient_checkpointing`
  - `gradient_accumulation_steps`
- Objectif :
  - réduire l’usage VRAM/RAM,
  - garantir la stabilité sur Colab

---

## 6. Évaluation

### 6.1 Métrique
- **ROUGE** :
  - ROUGE-1 (unigrammes)
  - ROUGE-2 (bigrams)
  - ROUGE-L / ROUGE-Lsum (structure globale)

### 6.2 Résultats clés (Llama 3.2-3B)
- Le modèle **PEFT** :
  - améliore la **cohérence locale** (ROUGE-2),
  - améliore légèrement la **structure globale** (ROUGE-L),
  - peut réduire légèrement le recouvrement lexical (ROUGE-1),
    ce qui est cohérent avec une génération plus **abstraite**.

👉 Les résultats montrent que **PEFT permet une adaptation ciblée et efficace**, même sur des modèles récents.

---

## 7. Enseignements principaux

- Le **fine-tuning complet** est souvent inutile et coûteux.
- **LoRA / QLoRA** permettent :
  - une réduction drastique des ressources nécessaires,
  - une grande flexibilité (changement d’adapter sans recharger le modèle).
- Les métriques relatives (en %) peuvent être trompeuses :
  - il est essentiel d’analyser aussi les **gains absolus**.
- ROUGE reste une métrique limitée :
  - une évaluation humaine ou sémantique serait complémentaire.

---

## 8. Limites et perspectives

### Limites
- Évaluation sur un petit nombre d’exemples (variance élevée).
- Dépendance à ROUGE (métrique lexicale).

### Perspectives
- Entraîner sur un plus grand sous-ensemble.
- Tester d’autres configurations LoRA (rank, modules ciblés).
- Comparer avec **Qwen 3 1.7B**.
- Ajouter des métriques sémantiques (BERTScore).

---

## 9. Conclusion
Ce projet démontre qu’il est possible de **fine-tuner efficacement des LLM modernes** avec des ressources limitées grâce à **PEFT / QLoRA**, tout en obtenant des gains mesurables sur une tâche de résumé.

Il constitue une base solide pour des projets industriels ou de recherche nécessitant une **adaptation rapide et peu coûteuse** de modèles de grande taille.

