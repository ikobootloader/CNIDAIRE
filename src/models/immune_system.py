import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
from src.config.settings import sim_config

@dataclass
class ImmuneParameters:
    """Paramètres du système immunitaire primitif des cnidaires"""
    # Capacités de base
    base_recognition_rate: float = 0.7    # Taux de reconnaissance des pathogènes
    base_response_strength: float = 1.0    # Force de la réponse immunitaire
    memory_duration: float = 300.0         # Durée de la mémoire immunitaire (sec)
    
    # Coûts énergétiques
    response_energy_cost: float = 0.05     # Coût énergétique par réponse
    maintenance_cost: float = 0.01         # Coût de maintenance par seconde
    
    # Seuils et limites
    activation_threshold: float = 0.3      # Seuil d'activation de la réponse
    max_concurrent_responses: int = 3      # Nombre max de réponses simultanées
    recovery_rate: float = 0.1            # Taux de récupération post-réponse

class ThreatType(Enum):
    """Types de menaces reconnues par le système immunitaire"""
    PATHOGEN = "pathogen"         # Agents pathogènes
    TOXIN = "toxin"              # Toxines
    PHYSICAL_DAMAGE = "damage"    # Dommages physiques
    FOREIGN_TISSUE = "foreign"    # Tissus étrangers

@dataclass
class ImmuneResponse:
    """Représente une réponse immunitaire active"""
    threat_type: ThreatType
    location: Tuple[float, float]  # Position dans l'organisme
    intensity: float               # Intensité de la réponse
    duration: float               # Durée d'activation
    start_time: float            # Temps de début
    
class ImmuneSystem:
    """
    Système immunitaire primitif des cnidaires, caractérisé par:
    - Reconnaissance non-spécifique des menaces
    - Réponses localisées
    - Mémoire immunitaire simple
    - Intégration avec le système nerveux
    """
    def __init__(self, params: Optional[ImmuneParameters] = None):
        self.params = params or ImmuneParameters()
        
        # État du système
        self.active_responses: List[ImmuneResponse] = []
        self.immune_memory: Dict[ThreatType, List[Tuple[float, float]]] = {
            threat_type: [] for threat_type in ThreatType
        }
        
        # Ressources
        self.energy_level = 1.0
        self.cell_reserves = 1.0
        
        # État d'activation
        self.is_active = False
        self.global_activation = 0.0
        self.local_activation_map = {}  # Position -> niveau d'activation
        
    def update(self, dt: float, nerve_activity: Optional[float] = None) -> Dict:
        """Met à jour l'état du système immunitaire"""
        # Consommation d'énergie basale
        self.energy_level = max(0.0, 
            self.energy_level - self.params.maintenance_cost * dt)
            
        # Mise à jour des réponses actives
        responses_to_remove = []
        for response in self.active_responses:
            # Vérification de la durée
            elapsed_time = response.duration - (time.time() - response.start_time)  # Correction ici
            if elapsed_time <= 0:
                responses_to_remove.append(response)
                continue
                
            # Consommation d'énergie pour la réponse
            energy_cost = (response.intensity * 
                         self.params.response_energy_cost * dt)
            if self.energy_level >= energy_cost:
                self.energy_level -= energy_cost
            else:
                response.intensity *= 0.5  # Réduction de l'intensité
            
            # Mise à jour de l'activation globale
            self.global_activation = max(
                self.global_activation,
                response.intensity
            )
                
        # Nettoyage des réponses terminées
        for response in responses_to_remove:
            self.active_responses.remove(response)
            
        # Mise à jour de la mémoire immunitaire
        self._update_immune_memory(dt)
            
        # Intégration avec le système nerveux si disponible
        if nerve_activity is not None:
            self.global_activation = max(
                self.global_activation,
                nerve_activity * 0.3  # Influence modérée du système nerveux
            )
            
        # Déclin naturel de l'activation globale
        self.global_activation = max(0, self.global_activation - 0.1 * dt)
                
        return {
            'active_responses': len(self.active_responses),
            'energy_level': self.energy_level,
            'global_activation': self.global_activation,
            'memory_strength': self._calculate_memory_strength()
        }

    def detect_threat(self, threat_type: ThreatType, 
                     position: Tuple[float, float], 
                     intensity: float) -> bool:
        """Détecte et évalue une menace potentielle"""
        # Vérification de la capacité de réponse
        if (len(self.active_responses) >= self.params.max_concurrent_responses or
            self.energy_level < self.params.response_energy_cost):
            return False
            
        # Calcul de la probabilité de détection
        detection_probability = self.params.base_recognition_rate
            
        # Bonus de la mémoire immunitaire
        memory_bonus = self._check_immune_memory(threat_type, position)
        detection_probability += memory_bonus
            
        # Influence de l'état d'activation global
        detection_probability *= (1.0 + self.global_activation)
            
        # Détection réussie ?
        if (np.random.random() < detection_probability and 
            intensity >= self.params.activation_threshold):
            # Ajout de la réponse avec le temps actuel
            response = ImmuneResponse(
                threat_type=threat_type,
                location=position,
                intensity=intensity,
                duration=30.0,  # Durée fixe de 30 secondes
                start_time=time.time()  # Temps actuel
            )
            self.active_responses.append(response)
            
            # Activation immédiate du système
            self.global_activation = max(self.global_activation, intensity)
            
            return True
                
        return False
        
    def _initiate_response(self, threat_type: ThreatType, 
                          position: Tuple[float, float], 
                          threat_intensity: float):
        """Initie une nouvelle réponse immunitaire"""
        # Calcul de l'intensité de la réponse
        response_intensity = (
            self.params.base_response_strength *
            (1.0 + self._check_immune_memory(threat_type, position)) *
            min(1.0, threat_intensity)
        )
        
        # Ajustement basé sur l'énergie disponible
        response_intensity *= min(1.0, self.energy_level * 2)
        
        # Durée de la réponse
        response_duration = 30.0 * (1.0 + threat_intensity)
        
        # Création de la réponse
        response = ImmuneResponse(
            threat_type=threat_type,
            location=position,
            intensity=response_intensity,
            duration=response_duration,
            start_time=0.0  # Sera mis à jour dans la boucle principale
        )
        
        # Ajout aux réponses actives
        self.active_responses.append(response)
        
        # Mise à jour de l'activation locale
        self._update_local_activation(position, response_intensity)
        
        # Ajout à la mémoire immunitaire
        self.immune_memory[threat_type].append((
            response_intensity,
            0.0  # Temps écoulé, sera mis à jour
        ))

    def _update_immune_memory(self, dt: float):
        """Met à jour la mémoire immunitaire"""
        for threat_type in ThreatType:
            # Mise à jour des temps
            updated_memory = []
            for intensity, elapsed_time in self.immune_memory[threat_type]:
                new_elapsed = elapsed_time + dt
                if new_elapsed < self.params.memory_duration:
                    updated_memory.append((intensity, new_elapsed))
                    
            self.immune_memory[threat_type] = updated_memory

    def _check_immune_memory(self, threat_type: ThreatType, 
                           position: Tuple[float, float]) -> float:
        """Vérifie la mémoire immunitaire pour un type de menace"""
        if not self.immune_memory[threat_type]:
            return 0.0
            
        # Calcul du bonus basé sur les réponses précédentes
        memory_strength = 0.0
        for intensity, elapsed_time in self.immune_memory[threat_type]:
            # Décroissance temporelle
            time_factor = 1.0 - (elapsed_time / self.params.memory_duration)
            memory_strength = max(memory_strength, intensity * time_factor)
            
        return memory_strength

    def _update_local_activation(self, position: Tuple[float, float], 
                               intensity: float):
        """Met à jour la carte d'activation locale"""
        # Discrétisation de la position pour la carte
        grid_pos = (int(position[0] / 10), int(position[1] / 10))
        
        # Mise à jour de l'activation locale
        current_activation = self.local_activation_map.get(grid_pos, 0.0)
        self.local_activation_map[grid_pos] = min(
            1.0, 
            current_activation + intensity * 0.5
        )
        
        # Propagation aux cellules voisines
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                neighbor_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                neighbor_activation = self.local_activation_map.get(neighbor_pos, 0.0)
                self.local_activation_map[neighbor_pos] = min(
                    1.0,
                    neighbor_activation + intensity * 0.2
                )

    def _calculate_memory_strength(self) -> float:
        """Calcule la force globale de la mémoire immunitaire"""
        total_strength = 0.0
        memory_count = 0
        
        for threat_type in ThreatType:
            for intensity, elapsed_time in self.immune_memory[threat_type]:
                time_factor = 1.0 - (elapsed_time / self.params.memory_duration)
                total_strength += intensity * time_factor
                memory_count += 1
                
        return total_strength / max(1, memory_count)
        
    def get_local_response(self, position: Tuple[float, float]) -> float:
        """Retourne le niveau de réponse immunitaire à une position donnée"""
        # Vérification des réponses actives dans la zone
        total_response = 0.0
        for response in self.active_responses:
            distance = np.sqrt(
                (position[0] - response.location[0])**2 +
                (position[1] - response.location[1])**2
            )
            if distance < 20.0:  # Rayon d'influence
                total_response += response.intensity * (1.0 - distance/20.0)
                
        return min(1.0, total_response)
        
    def transfer_immunity(self, other: 'ImmuneSystem', strength: float = 0.5):
        """Transfert une partie de la mémoire immunitaire (reproduction/bourgeonnement)"""
        for threat_type in ThreatType:
            for intensity, elapsed_time in self.immune_memory[threat_type]:
                if np.random.random() < strength:
                    # Transfert avec une certaine dégradation
                    transferred_intensity = intensity * np.random.uniform(0.6, 0.9)
                    other.immune_memory[threat_type].append(
                        (transferred_intensity, elapsed_time)
                    )