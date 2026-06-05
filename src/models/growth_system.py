from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
from enum import Enum
from src.config.settings import sim_config

class GrowthStage(Enum):
    """Stades de développement du cnidaire"""
    JUVENILE = "juvenile"     # Stade jeune
    ADOLESCENT = "adolescent" # Développement des systèmes
    MATURE = "mature"         # Pleine capacité
    SENESCENT = "senescent"   # Vieillissement

@dataclass
class GrowthParameters:
    """Paramètres contrôlant la croissance et le vieillissement"""
    # Durées des phases (en secondes)
    juvenile_duration: float = 100.0
    adolescent_duration: float = 200.0
    mature_duration: float = 500.0
    
    # Taux de croissance
    base_growth_rate: float = 0.01
    max_size: float = 200.0  # Augmenté à 200.0
    min_size: float = 50.0   # Ajout d'une taille minimale de départ
    
    # Facteurs de vieillissement
    aging_rate: float = 0.001
    damage_accumulation_rate: float = 0.005
    repair_efficiency: float = 0.8
    
    # Seuils métaboliques
    min_energy_for_growth: float = 0.4
    optimal_temperature: float = 20.0
    temperature_tolerance: float = 5.0

class GrowthSystem:
    """Gère la croissance, le développement et le vieillissement"""
    def __init__(self, params: Optional[GrowthParameters] = None):
        self.params = params or GrowthParameters()
        
        # État de développement
        self.age = 0.0
        self.size = 10.0  # Taille initiale
        self.stage = GrowthStage.JUVENILE
        self.development_progress = 0.0
        
        # Indicateurs de santé
        self.damage_level = 0.0
        self.repair_capacity = 1.0
        self.metabolic_rate = 1.0
        
        # Historique
        self.growth_history: List[Dict] = []
        
    def update(self, dt: float, conditions: Dict) -> Dict:
        """
        Met à jour l'état de croissance et développement
        
        Args:
            dt: Delta temps
            conditions: Dictionnaire des conditions environnementales et physiologiques
                {
                    'energy_level': float,  # Niveau d'énergie (0-1)
                    'temperature': float,   # Température ambiante
                    'nutrient_level': float # Disponibilité des nutriments (0-1)
                }
        """
        self.age += dt
        
        # Mise à jour du stade de développement
        self._update_stage()
        
        # Calcul des facteurs de croissance
        growth_factor = self._calculate_growth_factor(conditions)
        
        # Croissance si les conditions sont favorables
        if growth_factor > 0:
            self._grow(dt, growth_factor)
        
        # Vieillissement et accumulation de dommages
        self._age(dt)
        
        # Réparation des dommages
        self._repair(dt, conditions['energy_level'])
        
        # Enregistrement de l'historique
        self._record_state(conditions)
        
        return self._get_status()
        
    def _update_stage(self):
        """Met à jour le stade de développement basé sur l'âge"""
        # Vérification si nous sommes dans la phase juvénile
        if self.age < self.params.juvenile_duration:
            self.stage = GrowthStage.JUVENILE
            self.development_progress = self.age / self.params.juvenile_duration
            
        # Vérification pour le stade adolescent    
        elif self.age < (self.params.juvenile_duration + self.params.adolescent_duration):
            self.stage = GrowthStage.ADOLESCENT
            adolescent_age = self.age - self.params.juvenile_duration
            self.development_progress = adolescent_age / self.params.adolescent_duration
            
        # Vérification pour le stade mature    
        elif self.age < (self.params.juvenile_duration + 
                        self.params.adolescent_duration + 
                        self.params.mature_duration):
            self.stage = GrowthStage.MATURE
            self.development_progress = 1.0
            
        # Stade sénescent    
        else:
            self.stage = GrowthStage.SENESCENT
            self.development_progress = 1.0

        # Mettre à jour l'organisme parent avec le nouveau stade
        if hasattr(self, 'organism') and self.organism:
            self.organism.developmental_stage = self.stage.value
            
    def _calculate_growth_factor(self, conditions: Dict) -> float:
        """Calcule le facteur de croissance basé sur les conditions"""
        if conditions['energy_level'] < self.params.min_energy_for_growth:
            return 0.0
            
        # Influence de la température
        temp_effect = np.exp(
            -(conditions['temperature'] - self.params.optimal_temperature)**2 /
            (2 * self.params.temperature_tolerance**2)
        )
        
        # Facteur basé sur le stade de développement
        stage_factor = {
            GrowthStage.JUVENILE: 1.0,
            GrowthStage.ADOLESCENT: 0.7,
            GrowthStage.MATURE: 0.3,
            GrowthStage.SENESCENT: 0.1
        }[self.stage]
        
        # Facteur combiné
        growth_factor = (
            self.params.base_growth_rate *
            stage_factor *
            temp_effect *
            conditions['nutrient_level']
        )
        
        return max(0.0, growth_factor)
        
    def _grow(self, dt: float, growth_factor: float):
        """Applique la croissance"""
        if self.size >= self.params.max_size:
            return
            
        # Croissance avec ralentissement près de la taille maximale
        size_factor = 1.0 - (self.size / self.params.max_size)
        growth = growth_factor * dt * size_factor
        
        # Application de la croissance
        self.size = min(self.params.max_size, self.size + growth)
        
        # Ajustement du métabolisme
        self.metabolic_rate = 1.0 + (self.size / self.params.max_size) * 0.5
        
    def _age(self, dt: float):
        """Applique les effets du vieillissement"""
        # Accumulation naturelle de dommages
        base_damage = self.params.damage_accumulation_rate * dt
        
        # Facteurs augmentant les dommages
        damage_multiplier = 1.0
        if self.stage == GrowthStage.SENESCENT:
            damage_multiplier *= 2.0
        if self.metabolic_rate > 1.2:
            damage_multiplier *= self.metabolic_rate
            
        # Application des dommages
        self.damage_level = min(1.0, 
            self.damage_level + base_damage * damage_multiplier
        )
        
        # Réduction de la capacité de réparation avec l'âge
        if self.stage == GrowthStage.SENESCENT:
            self.repair_capacity = max(0.2, 
                self.repair_capacity - 0.01 * dt
            )
            
    def _repair(self, dt: float, energy_level: float):
        """Réparation des dommages cellulaires"""
        if energy_level < 0.2 or self.damage_level < 0.1:
            return
            
        # Calcul de la capacité de réparation
        repair_amount = (
            self.params.repair_efficiency *
            self.repair_capacity *
            energy_level *
            dt
        )
        
        # Application de la réparation
        self.damage_level = max(0.0,
            self.damage_level - repair_amount
        )
        
    def _record_state(self, conditions: Dict):
        """Enregistre l'état actuel dans l'historique"""
        state = {
            'age': self.age,
            'size': self.size,
            'stage': self.stage.value,
            'damage_level': self.damage_level,
            'repair_capacity': self.repair_capacity,
            'metabolic_rate': self.metabolic_rate,
            'conditions': conditions.copy()
        }
        
        self.growth_history.append(state)
        
        # Limite la taille de l'historique
        if len(self.growth_history) > 1000:
            self.growth_history.pop(0)
            
    def _get_status(self) -> Dict:
        """Retourne l'état actuel du système"""
        return {
            'age': self.age,
            'size': self.size,
            'stage': self.stage.value,
            'development_progress': self.development_progress,
            'damage_level': self.damage_level,
            'repair_capacity': self.repair_capacity,
            'metabolic_rate': self.metabolic_rate,
            'vitality': self._calculate_vitality()
        }
        
    def _calculate_vitality(self) -> float:
        """Calcule l'indice de vitalité global"""
        # Facteurs influençant la vitalité
        size_factor = self.size / self.params.max_size
        damage_factor = 1.0 - self.damage_level
        repair_factor = self.repair_capacity
        
        # Pondération selon le stade
        stage_weight = {
            GrowthStage.JUVENILE: 0.9,
            GrowthStage.ADOLESCENT: 1.0,
            GrowthStage.MATURE: 0.8,
            GrowthStage.SENESCENT: 0.6
        }[self.stage]
        
        vitality = (
            0.4 * size_factor +
            0.3 * damage_factor +
            0.3 * repair_factor
        ) * stage_weight
        
        return max(0.0, min(1.0, vitality))