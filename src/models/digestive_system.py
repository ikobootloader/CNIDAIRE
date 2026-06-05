import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from src.config.settings import sim_config

class DigestiveState(Enum):
    """États possibles du système digestif"""
    IDLE = "idle"               # Au repos
    INGESTING = "ingesting"     # Ingestion d'une proie
    DIGESTING = "digesting"     # Digestion en cours
    ABSORBING = "absorbing"     # Absorption des nutriments
    EXPELLING = "expelling"     # Expulsion des déchets

@dataclass
class GastricParameters:
    """Paramètres du système gastrovasculaire"""
    # Dimensions de la cavité gastrique
    cavity_radius: float = 25.0        # Rayon de la cavité gastrique
    wall_thickness: float = 3.0        # Épaisseur de la paroi gastrique
    
    # Paramètres physiologiques
    max_capacity: float = 100.0        # Capacité maximale en unités de nourriture
    digestion_rate: float = 0.05       # Taux de digestion par seconde
    absorption_rate: float = 0.08      # Taux d'absorption des nutriments
    enzyme_production_rate: float = 0.1 # Production d'enzymes digestives
    
    # Paramètres de contraction
    contraction_strength: float = 1.2   # Force des contractions digestives
    peristalsis_frequency: float = 0.2  # Fréquence des mouvements péristaltiques

class DigestiveCell:
    """Cellule gastrodermique avec capacités digestives"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.enzyme_level = 1.0
        self.nutrient_storage = 0.0
        self.activity = 0.0
        self.is_absorbing = False
        
    def update(self, dt: float, local_food_concentration: float):
        """Met à jour l'état de la cellule digestive"""
        # Production d'enzymes
        if local_food_concentration > 0.1:
            enzyme_production = dt * GastricParameters.enzyme_production_rate
            self.enzyme_level = min(1.0, self.enzyme_level + enzyme_production)
        
        # Digestion et absorption
        if local_food_concentration > 0 and self.enzyme_level > 0.1:
            digestion_amount = min(
                local_food_concentration,
                self.enzyme_level * GastricParameters.digestion_rate * dt
            )
            self.enzyme_level -= digestion_amount * 0.1
            self.nutrient_storage += digestion_amount * GastricParameters.absorption_rate
            self.is_absorbing = True
        else:
            self.is_absorbing = False
            
        # Activité cellulaire
        self.activity = (self.enzyme_level + self.nutrient_storage) / 2
        
        return {
            "digested_amount": digestion_amount if 'digestion_amount' in locals() else 0,
            "nutrient_stored": self.nutrient_storage
        }

class GastricCavity:
    """
    Cavité gastrovasculaire du cnidaire:
    - Digestion et distribution des nutriments
    - Mouvements péristaltiques
    - Absorption et circulation
    """
    def __init__(self, center_x: float, center_y: float, params: Optional[GastricParameters] = None):
        self.center_x = center_x
        self.center_y = center_y
        self.params = params or GastricParameters()
        
        # État physiologique
        self.current_state = DigestiveState.IDLE
        self.food_content = 0.0
        self.nutrient_level = 0.0
        self.waste_level = 0.0
        
        # Cellules digestives
        self.digestive_cells = self._initialize_digestive_cells()
        
        # État mécanique
        self.contraction_phase = 0.0
        self.peristaltic_wave = 0.0
        
        # Distribution des nutriments
        self.nutrient_distribution = np.zeros((20, 20))  # Grille de distribution
        
    def _initialize_digestive_cells(self) -> List[DigestiveCell]:
        """Initialise la couche de cellules digestives"""
        cells = []
        n_cells = int(2 * np.pi * self.params.cavity_radius / 2)  # Une cellule tous les 2 pixels
        
        for i in range(n_cells):
            angle = (2 * np.pi * i) / n_cells
            # Distribution sur la paroi de la cavité
            x = self.center_x + self.params.cavity_radius * np.cos(angle)
            y = self.center_y + self.params.cavity_radius * np.sin(angle)
            cells.append(DigestiveCell(x, y))
            
        return cells
        
    def update(self, dt: float):
        """Met à jour l'état du système digestif"""
        # Mise à jour de l'état péristaltique
        self._update_peristalsis(dt)
        
        # Distribution de la nourriture basée sur les mouvements péristaltiques
        food_distribution = self._compute_food_distribution()
        
        # Mise à jour des cellules digestives
        total_digested = 0
        total_nutrients = 0
        
        for cell in self.digestive_cells:
            # Calcul des indices normalisés pour la grille de distribution
            norm_x = int(((cell.x - self.center_x + self.params.cavity_radius) / 
                         (2 * self.params.cavity_radius)) * 19)  # 19 au lieu de 20
            norm_y = int(((cell.y - self.center_y + self.params.cavity_radius) / 
                         (2 * self.params.cavity_radius)) * 19)  # 19 au lieu de 20
            
            # S'assurer que les indices sont dans les limites
            norm_x = max(0, min(19, norm_x))
            norm_y = max(0, min(19, norm_y))
            
            local_food = food_distribution[norm_x, norm_y]
            
            result = cell.update(dt, local_food)
            total_digested += result["digested_amount"]
            total_nutrients += result["nutrient_stored"]
        
        # Mise à jour des niveaux globaux
        self.food_content = max(0, self.food_content - total_digested)
        self.nutrient_level += total_nutrients * dt
        self.waste_level += total_digested * 0.3 * dt  # 30% de déchets
        
        # Mise à jour de l'état digestif
        self._update_digestive_state()
    
    def _update_peristalsis(self, dt: float):
        """Met à jour les mouvements péristaltiques de la cavité"""
        # Progression de la phase péristaltique
        self.contraction_phase += dt * self.params.peristalsis_frequency * 2 * np.pi
        if self.contraction_phase >= 2 * np.pi:
            self.contraction_phase -= 2 * np.pi
            
        # Calcul de la vague péristaltique
        wave_speed = 1.0
        if self.current_state == DigestiveState.DIGESTING:
            wave_speed = 1.5  # Accélération pendant la digestion
        elif self.current_state == DigestiveState.INGESTING:
            wave_speed = 2.0  # Plus rapide pendant l'ingestion
            
        self.peristaltic_wave = np.sin(self.contraction_phase) * wave_speed
        
    def _compute_food_distribution(self) -> np.ndarray:
        """Calcule la distribution de la nourriture dans la cavité"""
        distribution = np.zeros((20, 20))
        
        # Centre de la distribution
        center = np.array([10, 10])
        
        # Effet de la vague péristaltique
        wave_direction = np.array([
            np.cos(self.contraction_phase),
            np.sin(self.contraction_phase)
        ])
        
        # Calcul de la distribution
        for i in range(20):
            for j in range(20):
                point = np.array([i, j])
                # Distance au centre
                dist_to_center = np.linalg.norm(point - center)
                if dist_to_center < 8:  # Intérieur de la cavité
                    # Influence de la vague péristaltique
                    wave_influence = np.dot(
                        (point - center) / max(1, dist_to_center),
                        wave_direction
                    )
                    
                    # Distribution de base avec effet de la vague
                    base_distribution = np.exp(-0.1 * dist_to_center)
                    wave_effect = 0.2 * (1 + self.peristaltic_wave * wave_influence)
                    
                    distribution[i, j] = base_distribution * wave_effect
        
        # Normalisation et application du contenu actuel
        distribution = distribution * (self.food_content / max(1e-6, distribution.sum()))
        return distribution
        
    def _update_digestive_state(self):
        """Met à jour l'état digestif basé sur les conditions actuelles"""
        previous_state = self.current_state
        
        if self.current_state == DigestiveState.IDLE:
            if self.food_content > self.params.max_capacity * 0.1:
                self.current_state = DigestiveState.DIGESTING
        
        elif self.current_state == DigestiveState.INGESTING:
            if self.food_content >= self.params.max_capacity * 0.9:
                self.current_state = DigestiveState.DIGESTING
            elif self.food_content <= 0:
                self.current_state = DigestiveState.IDLE
                
        elif self.current_state == DigestiveState.DIGESTING:
            if self.food_content <= self.params.max_capacity * 0.1:
                self.current_state = DigestiveState.ABSORBING
                
        elif self.current_state == DigestiveState.ABSORBING:
            if self.nutrient_level >= self.params.max_capacity * 0.8:
                self.current_state = DigestiveState.EXPELLING
            elif self.food_content <= 0.1:
                self.current_state = DigestiveState.IDLE
                
        elif self.current_state == DigestiveState.EXPELLING:
            if self.waste_level <= 0.1:
                self.current_state = DigestiveState.IDLE
                
        # Notification de changement d'état
        if self.current_state != previous_state:
            self._on_state_change(previous_state)
            
    def _on_state_change(self, previous_state: DigestiveState):
        """Gère les transitions entre états digestifs"""
        if self.current_state == DigestiveState.DIGESTING:
            # Activation de la production d'enzymes
            for cell in self.digestive_cells:
                cell.enzyme_level = min(1.0, cell.enzyme_level + 0.3)
                
        elif self.current_state == DigestiveState.ABSORBING:
            # Optimisation de l'absorption
            self.params.absorption_rate *= 1.2
            
        elif self.current_state == DigestiveState.EXPELLING:
            # Augmentation des contractions
            self.params.peristalsis_frequency *= 1.5
            
    def ingest_food(self, amount: float, position: Tuple[float, float] = None):
        """Ingère de la nourriture à une position donnée"""
        if self.food_content >= self.params.max_capacity:
            return False
            
        # Position par défaut au centre si non spécifiée
        if position is None:
            position = (self.center_x, self.center_y)
            
        # Vérification que la position est dans la cavité
        dist_to_center = np.sqrt(
            (position[0] - self.center_x)**2 + 
            (position[1] - self.center_y)**2
        )
        
        if dist_to_center > self.params.cavity_radius:
            return False
            
        # Ajout de la nourriture
        available_capacity = self.params.max_capacity - self.food_content
        ingested_amount = min(amount, available_capacity)
        self.food_content += ingested_amount
        
        # Changement d'état si nécessaire
        if self.current_state == DigestiveState.IDLE:
            self.current_state = DigestiveState.INGESTING
            
        return True
        
    def distribute_nutrients(self) -> Dict[str, float]:
        """Distribue les nutriments absorbés vers le reste de l'organisme"""
        if self.nutrient_level <= 0:
            return {"distributed": 0.0}
            
        distribution_amount = min(
            self.nutrient_level,
            self.params.absorption_rate * 2.0
        )
        
        self.nutrient_level -= distribution_amount
        
        return {
            "distributed": distribution_amount,
            "remaining": self.nutrient_level
        }