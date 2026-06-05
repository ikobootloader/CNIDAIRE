import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from src.config.settings import sim_config

class TissueType(Enum):
    """Types de tissus du cnidaire"""
    EPIDERMIS = "epidermis"      # Couche externe
    GASTRODERMIS = "gastrodermis"# Couche interne
    MESOGLEA = "mesoglea"        # Couche intermédiaire
    NERVE = "nerve"              # Tissu nerveux
    MUSCLE = "muscle"            # Tissu musculaire
    CNIDOCYTE = "cnidocyte"      # Cellules urticantes

@dataclass
class RegenerationParameters:
    """Paramètres du système de régénération"""
    # Taux de régénération de base (unités/seconde)
    base_regeneration_rate: float = 0.05
    
    # Coûts énergétiques
    energy_cost_per_cell: float = 0.1
    stem_cell_maintenance_cost: float = 0.02
    
    # Seuils et limites
    min_tissue_integrity: float = 0.2  # Seuil minimal pour la régénération
    max_regeneration_speed: float = 0.5 # Vitesse maximale de régénération
    
    # Facteurs environnementaux
    temperature_optimum: float = 20.0  # Température optimale (°C)
    temperature_tolerance: float = 5.0  # Tolérance à la variation
    
    # Capacités régénératives par type de tissu (0-1)
    tissue_regeneration_capacity: Dict[TissueType, float] = None
    
    def __post_init__(self):
        if self.tissue_regeneration_capacity is None:
            self.tissue_regeneration_capacity = {
                TissueType.EPIDERMIS: 0.9,      # Régénération rapide
                TissueType.GASTRODERMIS: 0.8,   # Bonne régénération
                TissueType.MESOGLEA: 0.6,       # Régénération modérée
                TissueType.NERVE: 0.4,          # Régénération limitée
                TissueType.MUSCLE: 0.7,         # Bonne régénération
                TissueType.CNIDOCYTE: 0.85      # Régénération rapide
            }

class TissueSection:
    """Représente une section de tissu régénérable"""
    def __init__(self, tissue_type: TissueType, initial_integrity: float = 1.0):
        self.tissue_type = tissue_type
        self.integrity = initial_integrity
        self.stem_cells = 1.0  # Niveau de cellules souches (0-1)
        self.regenerating = False
        self.age = 0.0
        
    def update(self, dt: float, params: RegenerationParameters, energy_available: float):
        """Met à jour l'état de la section de tissu"""
        self.age += dt
        
        # Dégradation naturelle basée sur l'âge
        natural_degradation = 0.001 * dt * (1.0 + self.age / 1000.0)
        self.integrity = max(0.0, self.integrity - natural_degradation)
        
        # Maintenance des cellules souches
        stem_cell_cost = params.stem_cell_maintenance_cost * dt
        if energy_available >= stem_cell_cost:
            energy_available -= stem_cell_cost
        else:
            self.stem_cells = max(0.0, self.stem_cells - 0.1 * dt)
            
class RegenerationSystem:
    """
    Système gérant la régénération tissulaire du cnidaire
    """
    def __init__(self, params: Optional[RegenerationParameters] = None):
        self.params = params or RegenerationParameters()
        
        # Carte des tissus par région
        self.tissue_map: Dict[str, List[TissueSection]] = {
            "body": self._initialize_body_tissues(),
            "tentacles": self._initialize_tentacle_tissues(),
            "gastric": self._initialize_gastric_tissues()
        }
        
        # État de la régénération
        self.regeneration_active = False
        self.current_focus = None  # Région actuellement en régénération
        self.stem_cell_reserves = 1.0
        self.regeneration_progress = {}
        
    def _initialize_body_tissues(self) -> List[TissueSection]:
        """Initialise les tissus du corps principal"""
        return [
            TissueSection(TissueType.EPIDERMIS),
            TissueSection(TissueType.MESOGLEA),
            TissueSection(TissueType.GASTRODERMIS),
            TissueSection(TissueType.MUSCLE),
            TissueSection(TissueType.NERVE)
        ]
        
    def _initialize_tentacle_tissues(self) -> List[TissueSection]:
        """Initialise les tissus des tentacules"""
        return [
            TissueSection(TissueType.EPIDERMIS),
            TissueSection(TissueType.NERVE),
            TissueSection(TissueType.MUSCLE),
            TissueSection(TissueType.CNIDOCYTE)
        ]
        
    def _initialize_gastric_tissues(self) -> List[TissueSection]:
        """Initialise les tissus gastriques"""
        return [
            TissueSection(TissueType.GASTRODERMIS),
            TissueSection(TissueType.MUSCLE),
            TissueSection(TissueType.NERVE)
        ]
        
    def update(self, dt: float, energy_available: float, environmental_conditions: Dict):
        """Met à jour l'état de régénération de tous les tissus"""
        # Vérification des conditions environnementales
        regen_factor = self._compute_environmental_factor(environmental_conditions)
        
        # Distribution de l'énergie disponible
        energy_per_tissue = energy_available / sum(
            len(tissues) for tissues in self.tissue_map.values()
        )
        
        # Mise à jour de chaque région
        for region, tissues in self.tissue_map.items():
            region_integrity = self._update_region(
                tissues, dt, energy_per_tissue, regen_factor
            )
            
            # Démarrage de la régénération si nécessaire
            if region_integrity < self.params.min_tissue_integrity:
                self._initiate_regeneration(region)
                
        # Mise à jour du processus de régénération actif
        if self.regeneration_active:
            self._update_active_regeneration(dt, energy_available)
            
    def _update_region(self, tissues: List[TissueSection], 
                      dt: float, energy_per_tissue: float, 
                      regen_factor: float) -> float:
        """Met à jour les tissus d'une région et retourne l'intégrité moyenne"""
        total_integrity = 0.0
        
        for tissue in tissues:
            # Mise à jour basique du tissu
            tissue.update(dt, self.params, energy_per_tissue)
            
            # Régénération si actif et énergie suffisante
            if tissue.regenerating and energy_per_tissue > 0:
                regen_rate = (
                    self.params.base_regeneration_rate * 
                    self.params.tissue_regeneration_capacity[tissue.tissue_type] *
                    regen_factor
                )
                
                # Calcul du coût et de la régénération possible
                energy_required = self.params.energy_cost_per_cell * regen_rate * dt
                if energy_required <= energy_per_tissue:
                    tissue.integrity = min(1.0, tissue.integrity + regen_rate * dt)
                    energy_per_tissue -= energy_required
                    
            total_integrity += tissue.integrity
            
        return total_integrity / len(tissues)
        
    def _compute_environmental_factor(self, conditions: Dict) -> float:
        """Calcule l'influence des conditions environnementales sur la régénération"""
        # Effet de la température
        temp_effect = np.exp(
            -(conditions.get('temperature', self.params.temperature_optimum) - 
              self.params.temperature_optimum)**2 / 
            (2 * self.params.temperature_tolerance**2)
        )
        
        # Effet de l'oxygène (si disponible)
        oxygen_effect = min(1.0, conditions.get('oxygen_level', 1.0))
        
        # Effet du pH (si disponible)
        ph = conditions.get('ph', 7.0)
        ph_effect = 1.0 - 0.2 * abs(ph - 7.0)
        
        return min(1.0, temp_effect * oxygen_effect * ph_effect)
        
    def _initiate_regeneration(self, region: str):
        """Démarre le processus de régénération pour une région"""
        if not self.regeneration_active:
            self.regeneration_active = True
            self.current_focus = region
            self.regeneration_progress[region] = 0.0
            
            # Activation de la régénération pour tous les tissus de la région
            for tissue in self.tissue_map[region]:
                tissue.regenerating = True
                
    def _update_active_regeneration(self, dt: float, energy_available: float):
        """Met à jour le processus de régénération actif"""
        if self.current_focus is None:
            return
            
        # Consommation de cellules souches
        stem_cell_cost = 0.01 * dt
        if self.stem_cell_reserves >= stem_cell_cost:
            self.stem_cell_reserves -= stem_cell_cost
            
            # Progression de la régénération
            progress_rate = 0.1 * dt * (energy_available / 100.0)  # Normalisé par rapport à l'énergie
            self.regeneration_progress[self.current_focus] += progress_rate
            
            # Vérification de la complétion
            if self.regeneration_progress[self.current_focus] >= 1.0:
                self._complete_regeneration(self.current_focus)
                
    def _complete_regeneration(self, region: str):
        """Finalise le processus de régénération pour une région"""
        # Désactivation de la régénération pour tous les tissus
        for tissue in self.tissue_map[region]:
            tissue.regenerating = False
            
        # Réinitialisation de l'état de régénération
        self.regeneration_active = False
        self.current_focus = None
        del self.regeneration_progress[region]

    def get_status(self) -> Dict:
        """Retourne l'état actuel du système de régénération"""
        return {
            "active": self.regeneration_active,
            "current_focus": self.current_focus,
            "stem_cell_reserves": self.stem_cell_reserves,
            "tissue_integrity": {
                region: sum(t.integrity for t in tissues) / len(tissues)
                for region, tissues in self.tissue_map.items()
            },
            "regeneration_progress": self.regeneration_progress.copy()
        }