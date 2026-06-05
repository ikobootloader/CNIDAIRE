import numpy as np
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from src.config.settings import sim_config

if TYPE_CHECKING:
    from .cnidarian_organism import CnidarianOrganism

class ReproductionMode(Enum):
    """Modes de reproduction asexuée disponibles"""
    BUDDING = "budding"           # Bourgeonnement
    FISSION = "fission"           # Fission binaire
    FRAGMENTATION = "fragmentation"# Fragmentation
    NONE = "none"                 # Pas de reproduction en cours

@dataclass
class ReproductionParameters:
    """Paramètres contrôlant la reproduction asexuée"""
    # Seuils énergétiques
    min_energy_for_reproduction: float = 0.8    # Énergie minimale nécessaire
    energy_cost_budding: float = 0.4           # Coût du bourgeonnement
    energy_cost_fission: float = 0.5           # Coût de la fission
    
    # Temporisation
    min_age_for_reproduction: float = 100.0     # Âge minimal (secondes)
    reproduction_cooldown: float = 200.0        # Temps entre reproductions
    budding_duration: float = 50.0             # Durée du bourgeonnement
    fission_duration: float = 30.0             # Durée de la fission
    
    # Facteurs de succès
    base_success_rate: float = 0.8             # Taux de succès de base
    size_factor: float = 0.2                   # Influence de la taille
    health_factor: float = 0.3                 # Influence de la santé
    
    # Paramètres de croissance
    initial_size_ratio: float = 0.3            # Taille initiale du bourgeon
    growth_rate: float = 0.01                  # Taux de croissance par seconde
    max_size_ratio: float = 0.8                # Taille maximale relative au parent

class BuddingProcess:
    """Gestion du processus de bourgeonnement"""
    def __init__(self, parent: 'CnidarianOrganism', position: Tuple[float, float]):
        self.parent = parent
        self.position = position
        self.elapsed_time = 0.0
        self.growth_progress = 0.0
        self.current_size = 0.0
        self.success_probability = 0.0
        self.tissue_differentiation = {}
        
    def update(self, dt: float, params: ReproductionParameters) -> bool:
        """Met à jour le processus de bourgeonnement"""
        # Croissance du bourgeon
        growth_increment = params.growth_rate * dt
        self.growth_progress = min(1.0, self.growth_progress + growth_increment)
        self.elapsed_time += dt
        
        # Mise à jour de la taille
        target_size = params.initial_size_ratio * (
            1.0 + (params.max_size_ratio - params.initial_size_ratio) * 
            self.growth_progress
        )
        self.current_size += (target_size - self.current_size) * 0.1
        
        # Différenciation des tissus
        self._update_tissue_differentiation(dt)
        
        # Vérification de la complétion
        return self.elapsed_time >= params.budding_duration

    def _update_tissue_differentiation(self, dt: float):
        """Met à jour la différenciation des différents tissus"""
        tissue_types = ['epidermis', 'gastrodermis', 'mesoglea', 'nerve', 'muscle']
        
        for tissue in tissue_types:
            if tissue not in self.tissue_differentiation:
                self.tissue_differentiation[tissue] = 0.0
                
            # Différenciation progressive
            target_progress = self.growth_progress
            current_progress = self.tissue_differentiation[tissue]
            
            # Les tissus se développent à des vitesses différentes
            rate_multiplier = {
                'epidermis': 1.2,    # Plus rapide
                'gastrodermis': 1.0,
                'mesoglea': 0.8,     # Plus lent
                'nerve': 0.7,        # Plus lent
                'muscle': 0.9
            }.get(tissue, 1.0)
            
            self.tissue_differentiation[tissue] = min(
                1.0,
                current_progress + 0.05 * rate_multiplier * dt
            )

class FissionProcess:
    """Gestion du processus de fission"""
    def __init__(self, parent: 'CnidarianOrganism', axis_angle: float):
        self.parent = parent
        self.axis_angle = axis_angle  # Angle de l'axe de fission
        self.elapsed_time = 0.0
        self.separation_progress = 0.0
        self.tissue_integrity = {'anterior': 1.0, 'posterior': 1.0}
        
    def update(self, dt: float, params: ReproductionParameters) -> bool:
        """Met à jour le processus de fission"""
        self.elapsed_time += dt
        # Progression de la séparation
        self.separation_progress = min(
            1.0,
            self.separation_progress + dt / params.fission_duration
        )
        
        # Mise à jour de l'intégrité tissulaire
        self._update_tissue_integrity(dt)
        
        # Vérification de la complétion
        return self.separation_progress >= 1.0
        
    def _update_tissue_integrity(self, dt: float):
        """Met à jour l'intégrité tissulaire pendant la fission"""
        # La fission réduit temporairement l'intégrité
        damage = 0.1 * dt
        for part in self.tissue_integrity:
            self.tissue_integrity[part] = max(
                0.5,  # Intégrité minimale
                self.tissue_integrity[part] - damage
            )
            
            # Régénération simultanée
            regen = 0.05 * dt * (1.0 - self.separation_progress)
            self.tissue_integrity[part] = min(
                1.0,
                self.tissue_integrity[part] + regen
            )
            
class ReproductionSystem:
    """
    Système gérant la reproduction asexuée du cnidaire
    """
    def __init__(self, organism: 'CnidarianOrganism', 
                 params: Optional[ReproductionParameters] = None):
        self.organism = organism
        self.params = params or ReproductionParameters()
        
        # État du système
        self.current_mode = ReproductionMode.NONE
        self.time_since_last_reproduction = self.params.reproduction_cooldown
        self.active_process = None
        
        # Statistiques
        self.reproduction_history = []
        self.offspring_count = {
            ReproductionMode.BUDDING: 0,
            ReproductionMode.FISSION: 0,
            ReproductionMode.FRAGMENTATION: 0
        }
        
    def update(self, dt: float) -> Optional['CnidarianOrganism']:
        """Met à jour le système de reproduction et retourne un nouvel organisme si créé"""
        self.time_since_last_reproduction += dt

        if self.active_process is None:
            # Vérification pour démarrer une nouvelle reproduction
            if self._can_reproduce():
                self._initiate_reproduction()
            return None
            
        # Mise à jour du processus en cours
        if self.current_mode == ReproductionMode.BUDDING:
            if self.active_process.update(dt, self.params):
                return self._complete_budding()
                
        elif self.current_mode == ReproductionMode.FISSION:
            if self.active_process.update(dt, self.params):
                return self._complete_fission()
                
        return None
        
    def _can_reproduce(self) -> bool:
        """Vérifie si l'organisme peut se reproduire"""
        # Vérification du temps écoulé depuis la dernière reproduction
        if self.time_since_last_reproduction < self.params.reproduction_cooldown:
            return False
            
        # Vérification de l'âge
        if self.organism.age < self.params.min_age_for_reproduction:
            return False
            
        # Vérification de l'énergie
        if self.organism.energy < self.params.min_energy_for_reproduction:
            return False
            
        # Vérification de la santé
        if self.organism.health < 0.5:
            return False
            
        return True
        
    def _initiate_reproduction(self):
        """Démarre un nouveau processus de reproduction"""
        # Sélection du mode de reproduction
        mode = self._select_reproduction_mode()
        self.current_mode = mode
        
        if mode == ReproductionMode.BUDDING:
            # Démarrage du bourgeonnement
            bud_position = self._calculate_bud_position()
            self.active_process = BuddingProcess(self.organism, bud_position)
            self.organism.energy -= self.params.energy_cost_budding
            
        elif mode == ReproductionMode.FISSION:
            # Démarrage de la fission
            fission_angle = self._calculate_fission_angle()
            self.active_process = FissionProcess(self.organism, fission_angle)
            self.organism.energy -= self.params.energy_cost_fission
            
        self.time_since_last_reproduction = 0.0
        
    def _select_reproduction_mode(self) -> ReproductionMode:
        """Sélectionne le mode de reproduction le plus approprié"""
        # Facteurs influençant la décision
        size_factor = self.organism.size / 100.0  # Normalisé
        energy_factor = self.organism.energy / self.params.min_energy_for_reproduction
        health_factor = self.organism.health
        
        # Scores pour chaque mode
        scores = {
            ReproductionMode.BUDDING: (
                0.7 * energy_factor + 
                0.2 * health_factor + 
                0.1 * size_factor
            ),
            ReproductionMode.FISSION: (
                0.5 * size_factor + 
                0.3 * energy_factor + 
                0.2 * health_factor
            ),
            ReproductionMode.FRAGMENTATION: (
                0.4 * health_factor + 
                0.3 * size_factor + 
                0.3 * energy_factor
            )
        }
        
        # Sélection du mode avec le meilleur score
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def _calculate_bud_position(self) -> Tuple[float, float]:
        """Calcule la position optimale pour le bourgeonnement"""
        # Position relative au corps du parent
        angle = np.random.uniform(0, 2 * np.pi)
        distance = self.organism.radius * 1.2
        
        rel_x = distance * np.cos(angle)
        rel_y = distance * np.sin(angle)
        
        return (self.organism.x + rel_x, self.organism.y + rel_y)
        
    def _calculate_fission_angle(self) -> float:
        """Calcule l'angle optimal pour la fission"""
        # Par défaut, perpendiculaire à l'axe principal
        return self.organism.orientation + np.pi/2 + np.random.normal(0, 0.1)
        
    def _complete_budding(self) -> 'CnidarianOrganism':
        """Finalise le processus de bourgeonnement"""
        # Création du nouvel organisme
        from .cnidarian_organism import CnidarianOrganism
        
        bud = CnidarianOrganism(
            x=self.active_process.position[0],
            y=self.active_process.position[1]
        )
        
        # Transfert des propriétés héritées
        bud.set_size(self.organism.size * self.active_process.current_size)
        bud.energy = self.organism.energy * 0.3
        bud.orientation = self.organism.orientation + np.random.normal(0, 0.2)
        
        # Mise à jour des statistiques
        self.offspring_count[ReproductionMode.BUDDING] += 1
        self._record_reproduction_event(ReproductionMode.BUDDING)
        
        # Réinitialisation du système
        self.current_mode = ReproductionMode.NONE
        self.active_process = None
        
        return bud
        
    def _complete_fission(self) -> 'CnidarianOrganism':
        """Finalise le processus de fission"""
        # Création du nouvel organisme
        from .cnidarian_organism import CnidarianOrganism
        
        # Position du nouvel organisme
        separation_distance = self.organism.radius * 2
        new_x = self.organism.x + separation_distance * np.cos(self.active_process.axis_angle)
        new_y = self.organism.y + separation_distance * np.sin(self.active_process.axis_angle)
        
        offspring = CnidarianOrganism(x=new_x, y=new_y)
        
        # Partage équitable des ressources
        offspring.set_size(self.organism.size * 0.5)
        self.organism.set_size(self.organism.size * 0.5)
        
        offspring.energy = self.organism.energy * 0.5
        self.organism.energy *= 0.5
        
        # Mise à jour des statistiques
        self.offspring_count[ReproductionMode.FISSION] += 1
        self._record_reproduction_event(ReproductionMode.FISSION)
        
        # Réinitialisation du système
        self.current_mode = ReproductionMode.NONE
        self.active_process = None
        
        return offspring
        
    def _record_reproduction_event(self, mode: ReproductionMode):
        """Enregistre un événement de reproduction dans l'historique"""
        event = {
            'time': self.organism.age,
            'mode': mode,
            'parent_size': self.organism.size,
            'parent_energy': self.organism.energy,
            'parent_health': self.organism.health
        }
        self.reproduction_history.append(event)
        
        # Limite la taille de l'historique
        if len(self.reproduction_history) > 100:
            self.reproduction_history.pop(0)
