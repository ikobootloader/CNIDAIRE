# Changelog

## 2026-06-03

- Correction de `src/models/cnidarian_organism.py` pour initialiser correctement `radius` et `size`.
- Suppression de pseudo-propriétés déclarées dans `__init__` qui empêchaient le fonctionnement normal de l'organisme.
- Correction d'une division NumPy sur vecteur entier dans la réponse défensive aux menaces.
- Intégration effective des systèmes de régénération, reproduction et collisions dans `CnidarianOrganism`.
- Réparation de `Environment`, `FeedingSystem` et `InteractionSystem` pour permettre une simulation multi-organismes cohérente.
- Ajout d'un orchestrateur principal `src/simulation/cnidarian_simulation.py` et d'un point d'entrée `src/main.py`.
- Ajout de tests ciblés pour l'environnement, l'alimentation, les collisions, la reproduction et l'orchestrateur.
- Amélioration du rendu pour mieux percevoir le déplacement : fond marin plus vivant, proies visibles, vecteurs de courant et traînées de mouvement.
- Renforcement de la lisibilité biomécanique de la nage : phase de pulsation, propulsion plus rythmée, ombrelle légèrement contractile et sillage pulsé.
- Correction des défauts d'intégration relevés en revue : plus de double update nerveux/immunitaire, retrait des proies seulement après ingestion réelle, pilotage non binaire des tentacules et overlay rendu dans le bon ordre.
- Ajout d'un comportement de nage plus finalisé : orientation vers la proie la plus proche, alternance effort/relâchement et dérive plus crédible sous l'effet du courant.
