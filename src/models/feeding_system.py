import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from .tentacles import Tentacle
from .digestive_system import GastricCavity, DigestiveState
from src.config.settings import sim_config

class PreyState(Enum):
    """États possibles d'une proie capturée"""
    FREE = "free"               # Non capturée
    CAPTURED = "captured"       # Capturée par les tentacules
    PARALYZED = "paralyzed"     # Paralysée par les cnidocytes
    TRANSFERRING = "transferring"# En cours de transfert vers la bouche
    INGESTED = "ingested"       # Ingérée dans la cavité gastrique

@dataclass
class PreyParameters:
    """Paramètres définissant une proie"""
    size: float                 # Taille de la proie
    energy_content: float       # Contenu énergétique
    resistance: float           # Résistance aux cnidocytes
    escape_strength: float      # Force d'échappement
    digestion_difficulty: float # Difficulté de digestion (1-10)

class FeedingSystem:
    """
    Système coordonnant la capture et l'ingestion des proies
    entre les tentacules et le système digestif
    """
    def __init__(self, 
                 tentacles: List[Tentacle], 
                 gastric_cavity: GastricCavity,
                 mouth_position: Tuple[float, float]):
        self.tentacles = tentacles
        self.gastric_cavity = gastric_cavity
        self.mouth_position = mouth_position
        
        # Gestion des proies
        self.captured_prey: Dict[int, Tuple[PreyState, PreyParameters, Tuple[float, float], Optional[int]]] = {}
        self.next_prey_id = 0
        self.ingested_world_prey_ids: List[int] = []
        
        # État du système
        self.is_feeding = False
        self.feeding_intensity = 0.0
        self.successful_captures = 0
        self.failed_captures = 0
        
        # Configuration des zones
        self.capture_radius = max(t.params.base_length for t in tentacles) * 1.2
        self.mouth_radius = 10.0
        
    def update(self, dt: float):
        """Met à jour le processus de capture et d'ingestion"""
        # Mise à jour de l'état de chaque proie
        prey_to_remove = []
        self.ingested_world_prey_ids.clear()
        
        for prey_id in list(self.captured_prey.keys()):
            state, params, position, _ = self._get_prey_entry(prey_id)
            if state == PreyState.CAPTURED:
                if self._check_prey_paralyzed(prey_id):
                    self._paralyze_prey(prey_id)
                    
            elif state == PreyState.PARALYZED:
                success = self._move_prey_to_mouth(prey_id, dt)
                if success:
                    self._start_ingestion(prey_id)
                    
            elif state == PreyState.TRANSFERRING:
                if self._check_prey_at_mouth(prey_id):
                    self._complete_ingestion(prey_id)
                    prey_to_remove.append(prey_id)
                    
        # Nettoyage des proies ingérées
        for prey_id in prey_to_remove:
            del self.captured_prey[prey_id]
            
        # Mise à jour de l'état global
        self.is_feeding = len(self.captured_prey) > 0
        self.feeding_intensity = len(self.captured_prey) / max(1, len(self.tentacles))
        
    def attempt_prey_capture(self, prey_position: Tuple[float, float], 
                           prey_params: PreyParameters,
                           world_prey_id: Optional[int] = None) -> bool:
        """Tente de capturer une nouvelle proie"""
        # Vérification de la capacité de capture
        if self.gastric_cavity.food_content >= self.gastric_cavity.params.max_capacity:
            return False
            
        # Vérification de la position de la proie
        distance_to_center = np.sqrt(
            (prey_position[0] - self.mouth_position[0])**2 +
            (prey_position[1] - self.mouth_position[1])**2
        )
        
        if distance_to_center > self.capture_radius:
            return False
            
        # Recherche du tentacule le plus proche
        closest_tentacle = min(
            self.tentacles,
            key=lambda t: self._distance_to_tentacle(prey_position, t)
        )
        
        # Tentative de capture
        capture_success = self._try_capture_with_tentacle(
            closest_tentacle, 
            prey_position, 
            prey_params
        )
        
        if capture_success:
            self.captured_prey[self.next_prey_id] = (
                PreyState.CAPTURED,
                prey_params,
                prey_position,
                world_prey_id
            )
            self.next_prey_id += 1
            self.successful_captures += 1
            return True
        else:
            self.failed_captures += 1
            return False
            
    def _distance_to_tentacle(self, position: Tuple[float, float], 
                            tentacle: Tentacle) -> float:
        """Calcule la distance minimale entre une position et un tentacule"""
        min_distance = float('inf')
        
        for segment in tentacle.segments:
            # Distance au segment
            segment_start = (segment.base_x, segment.base_y)
            segment_end = segment.tip_position
            
            # Calcul de la distance point-segment
            min_distance = min(
                min_distance,
                self._point_segment_distance(position, segment_start, segment_end)
            )
            
        return min_distance
        
    def _point_segment_distance(self, 
                              point: Tuple[float, float],
                              segment_start: Tuple[float, float],
                              segment_end: Tuple[float, float]) -> float:
        """Calcule la distance minimale entre un point et un segment"""
        px, py = point
        x1, y1 = segment_start
        x2, y2 = segment_end
        
        # Vecteur du segment
        dx = x2 - x1
        dy = y2 - y1
        
        # Si le segment est un point
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
            
        # Projection du point sur la ligne du segment
        t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
        
        if t < 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        elif t > 1:
            return np.sqrt((px - x2)**2 + (py - y2)**2)
        else:
            # Point projeté sur le segment
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
            
    def _try_capture_with_tentacle(self, 
                                 tentacle: Tentacle,
                                 prey_position: Tuple[float, float],
                                 prey_params: PreyParameters) -> bool:
        """Tente une capture avec un tentacule spécifique"""
        # Vérification de l'énergie du tentacule
        if tentacle.energy < 0.3:
            return False
            
        # Calcul du nombre de cnidocytes activés
        active_cnidocytes = sum(
            1 for point in tentacle.get_prey_capture_points()
            if self._point_segment_distance(prey_position, point, point) < 5.0
        )
        
        # Probabilité de capture basée sur plusieurs facteurs
        capture_probability = (
            (active_cnidocytes / 10.0) *        # Nombre de cnidocytes
            (1.0 - prey_params.resistance) *     # Résistance de la proie
            (1.0 - prey_params.size / 50.0) *   # Taille relative
            tentacle.energy                      # Énergie du tentacule
        )
        
        return np.random.random() < capture_probability
    
    def _check_prey_paralyzed(self, prey_id: int) -> bool:
        """Vérifie si une proie est suffisamment paralysée pour être transférée"""
        _, params, position, _ = self._get_prey_entry(prey_id)
        
        # Calcul de la force d'échappement résiduelle
        paralysis_duration = 0
        for tentacle in self.tentacles:
            # Vérification des cnidocytes autour de la proie
            nearby_cnidocytes = [
                cnido for segment in tentacle.segments
                for cnido in segment.cnidocytes
                if cnido.has_fired and 
                self._point_distance(position, (cnido.x, cnido.y)) < 10.0
            ]
            paralysis_duration += len(nearby_cnidocytes)
            
        escape_probability = (
            params.escape_strength * 
            np.exp(-0.1 * paralysis_duration) * 
            (1.0 - params.resistance)
        )
        
        return np.random.random() > escape_probability

    def _paralyze_prey(self, prey_id: int):
        """Change l'état d'une proie en paralysée"""
        _, params, position, world_prey_id = self._get_prey_entry(prey_id)
        self.captured_prey[prey_id] = (PreyState.PARALYZED, params, position, world_prey_id)
        
        # Stimulation du système nerveux pour initier le transfert
        for tentacle in self.tentacles:
            tentacle.local_nerve_net.stimulate_region(
                *position,
                radius=15.0,
                strength=0.8
            )

    def _move_prey_to_mouth(self, prey_id: int, dt: float) -> bool:
        """Déplace une proie paralysée vers la bouche"""
        _, params, position, world_prey_id = self._get_prey_entry(prey_id)
        
        # Vecteur vers la bouche
        dx = self.mouth_position[0] - position[0]
        dy = self.mouth_position[1] - position[1]
        distance_to_mouth = np.sqrt(dx**2 + dy**2)
        
        if distance_to_mouth < self.mouth_radius:
            return True
            
        # Calcul du mouvement
        move_speed = 20.0 * dt * (1.0 - params.size/50.0)  # Vitesse inversement proportionnelle à la taille
        normalized_dx = dx / distance_to_mouth
        normalized_dy = dy / distance_to_mouth
        
        # Nouvelle position
        new_x = position[0] + normalized_dx * move_speed
        new_y = position[1] + normalized_dy * move_speed
        
        # Mise à jour de la position
        self.captured_prey[prey_id] = (PreyState.TRANSFERRING, params, (new_x, new_y), world_prey_id)
        
        # Coordination des tentacules pour le transfert
        self._coordinate_tentacle_movement(prey_id)
        
        return False
        
    def _coordinate_tentacle_movement(self, prey_id: int):
        """Coordonne les tentacules pour le transfert de la proie"""
        _, _, position, _ = self._get_prey_entry(prey_id)
        
        for tentacle in self.tentacles:
            # Distance entre le tentacule et la proie
            tentacle_base = (tentacle.base_x, tentacle.base_y)
            distance = self._point_distance(position, tentacle_base)
            
            if distance < tentacle.params.base_length:
                # Calcul de l'angle vers la bouche
                angle_to_mouth = np.arctan2(
                    self.mouth_position[1] - position[1],
                    self.mouth_position[0] - position[0]
                )
                
                # Ajustement des segments du tentacule
                for i, segment in enumerate(tentacle.segments):
                    # Angle progressif vers la bouche
                    target_angle = angle_to_mouth + np.pi/6 * np.sin(i/len(tentacle.segments))
                    segment.target_angle = target_angle
                    
                # Stimulation du réseau nerveux local
                tentacle.local_nerve_net.stimulate_region(
                    *position,
                    radius=10.0,
                    strength=0.5
                )

    def _check_prey_at_mouth(self, prey_id: int) -> bool:
        """Vérifie si une proie est arrivée à la bouche"""
        _, _, position, _ = self._get_prey_entry(prey_id)
        distance_to_mouth = self._point_distance(position, self.mouth_position)
        return distance_to_mouth < self.mouth_radius

    def _start_ingestion(self, prey_id: int):
        """Marque une proie comme prête à être ingérée."""
        _, params, _, world_prey_id = self._get_prey_entry(prey_id)
        self.captured_prey[prey_id] = (
            PreyState.TRANSFERRING,
            params,
            self.mouth_position,
            world_prey_id
        )
        self.gastric_cavity.current_state = DigestiveState.INGESTING

    def _complete_ingestion(self, prey_id: int):
        """Termine l'ingestion d'une proie dans la cavité gastrique"""
        _, params, _, world_prey_id = self._get_prey_entry(prey_id)
        
        # Conversion de la proie en nourriture pour la cavité gastrique
        food_amount = params.energy_content * (1.0 - 0.1 * params.digestion_difficulty)
        
        # Tentative d'ingestion
        ingestion_success = self.gastric_cavity.ingest_food(
            amount=food_amount,
            position=self.mouth_position
        )
        
        if ingestion_success:
            # Signal de succès aux tentacules
            for tentacle in self.tentacles:
                # Récompense pour le réseau nerveux
                tentacle.local_nerve_net.stimulate_region(
                    *self.mouth_position,
                    radius=20.0,
                    strength=0.3
                )
                
            # Récupération d'énergie pour les tentacules impliqués
            self._distribute_capture_energy(params.energy_content * 0.1)
            if world_prey_id is not None:
                self.ingested_world_prey_ids.append(world_prey_id)

    def _distribute_capture_energy(self, energy_amount: float):
        """Distribue l'énergie de capture entre les tentacules"""
        energy_per_tentacle = energy_amount / len(self.tentacles)
        for tentacle in self.tentacles:
            tentacle.energy = min(1.0, tentacle.energy + energy_per_tentacle)

    def _point_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calcule la distance entre deux points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _get_prey_entry(self, prey_id: int) -> Tuple[PreyState, PreyParameters, Tuple[float, float], Optional[int]]:
        """Normalise les anciens et nouveaux formats de stockage des proies capturées."""
        entry = self.captured_prey[prey_id]
        if len(entry) == 3:
            state, params, position = entry
            return state, params, position, None
        return entry

    def pop_ingested_world_prey_ids(self) -> List[int]:
        """Retourne puis vide les identifiants monde des proies réellement ingérées."""
        ingested = list(self.ingested_world_prey_ids)
        self.ingested_world_prey_ids.clear()
        return ingested
