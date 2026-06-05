import numpy as np
from dataclasses import dataclass
import pygame
from typing import List, Tuple, Dict, Optional
from .nervous_system import DiffuseNerveNet
from src.config.settings import sim_config

@dataclass
class TentacleParameters:
    """Paramètres physiques et biologiques des tentacules"""
    # Paramètres structurels
    base_length: float = 40.0
    n_segments: int = 8
    segment_elasticity: float = 0.8
    
    # Paramètres des cnidocytes (cellules urticantes)
    cnidocyte_density: float = 0.3
    cnidocyte_recharge_time: float = 5.0
    discharge_threshold: float = 0.7
    
    # Paramètres musculaires
    contraction_strength: float = 1.2
    relaxation_rate: float = 0.3
    max_contraction: float = 0.5

class Cnidocyte:
    """Cellule urticante (cnidocyte) caractéristique des cnidaires"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.is_charged = True
        self.recharge_timer = 0.0
        self.has_fired = False
        self.sensitivity = 1.0
        
    def update(self, dt: float, stimulus: float = 0.0) -> bool:
        """Met à jour l'état de la cellule et retourne si elle se décharge"""
        if not self.is_charged:
            self.recharge_timer += dt
            if self.recharge_timer >= TentacleParameters.cnidocyte_recharge_time:
                self.is_charged = True
                self.recharge_timer = 0.0
                self.has_fired = False
                return False
                
        if self.is_charged and stimulus * self.sensitivity >= TentacleParameters.discharge_threshold:
            self.is_charged = False
            self.has_fired = True
            return True
            
        return False

class TentacleSegment:
    """Segment individuel d'un tentacule avec muscles et cnidocytes"""
    def __init__(self, base_x: float, base_y: float, length: float, angle: float, segment_index: int):
        self.base_x = base_x
        self.base_y = base_y
        self.length = length
        self.angle = angle
        self.segment_index = segment_index  # Ajout de l'index du segment
        
        # État physique
        self.target_angle = angle
        self.contraction = 0.0
        self.velocity = 0.0
        
        # Cnidocytes distribués le long du segment
        self.cnidocytes = self._initialize_cnidocytes()
        
        # État musculaire
        self.muscle_tension = 0.0
        self.neural_input = 0.0
        
    def _initialize_cnidocytes(self) -> List[Cnidocyte]:
        """Initialise les cnidocytes le long du segment"""
        cnidocytes = []
        n_cnidocytes = int(self.length * TentacleParameters.cnidocyte_density)
        
        for i in range(n_cnidocytes):
            relative_pos = i / max(1, n_cnidocytes - 1)
            x = self.base_x + self.length * relative_pos * np.cos(self.angle)
            y = self.base_y + self.length * relative_pos * np.sin(self.angle)
            cnidocytes.append(Cnidocyte(x, y))
            
        return cnidocytes
        
    def update(self, dt: float, neural_activity: float):
        """Met à jour l'état physique et biologique du segment"""
        # Mise à jour de la tension musculaire
        self.muscle_tension += (neural_activity - self.muscle_tension) * dt * 5.0
        
        # Application des forces physiques
        target_contraction = self.muscle_tension * TentacleParameters.contraction_strength
        self.contraction += (target_contraction - self.contraction) * dt * 2.0
        
        # Mise à jour de l'angle avec inertie et élasticité
        angular_force = (self.target_angle - self.angle) * TentacleParameters.segment_elasticity
        self.velocity += angular_force * dt
        self.velocity *= 0.9  # Amortissement
        self.angle += self.velocity * dt
        
        # Mise à jour des cnidocytes
        local_stimulus = neural_activity * 1.2  # Amplification du stimulus
        for cnidocyte in self.cnidocytes:
            cnidocyte.update(dt, local_stimulus)
            
    @property
    def tip_position(self) -> Tuple[float, float]:
        """Calcule la position de l'extrémité du segment"""
        contracted_length = self.length * (1.0 - self.contraction * TentacleParameters.max_contraction)
        tip_x = self.base_x + contracted_length * np.cos(self.angle)
        tip_y = self.base_y + contracted_length * np.sin(self.angle)
        return tip_x, tip_y

class Tentacle:
    """Tentacule complet avec segments articulés et système nerveux local"""
    def __init__(self, base_x: float, base_y: float, base_angle: float):
        self.base_x = base_x
        self.base_y = base_y
        self.base_angle = base_angle
        
        # Paramètres
        self.params = TentacleParameters()
        
        # Création des segments
        self.segments = self._initialize_segments()
        
        # Réseau nerveux local
        self.local_nerve_net = DiffuseNerveNet(
            n_neurons=20,  # Réseau plus petit pour le tentacule
            radius=self.params.base_length * 0.2
        )
        
        # État physiologique
        self.sensitivity = 1.0
        self.energy = 1.0
        
    def _initialize_segments(self) -> List[TentacleSegment]:
        """Initialise les segments du tentacule"""
        segments = []
        segment_length = self.params.base_length / self.params.n_segments
        
        current_x = self.base_x
        current_y = self.base_y
        current_angle = self.base_angle
        
        for i in range(self.params.n_segments):
            segment = TentacleSegment(
                current_x, 
                current_y, 
                segment_length, 
                current_angle,
                segment_index=i  # Ajout de l'index lors de la création
            )
            segments.append(segment)
            
            # Position de base du prochain segment
            current_x, current_y = segment.tip_position
            current_angle += np.random.normal(0, 0.1)  # Légère variation d'angle
            
        return segments
    
    def update(self, dt: float, global_contraction: float, stimuli: Optional[Dict[str, float]] = None):
        """
        Met à jour l'état du tentacule en fonction des contractions globales 
        et des stimuli locaux
        """
        # Mise à jour du réseau nerveux local
        local_activity = self.local_nerve_net.update(dt)
        
        # Traitement des stimuli
        if stimuli:
            self._process_stimuli(stimuli)
        
        # Propagation de l'activité le long des segments
        for i, segment in enumerate(self.segments):
            # Calcul de l'activité neurale locale pour ce segment
            segment_position = segment.base_x, segment.base_y
            local_neural_input = self.local_nerve_net.get_local_activity(
                *segment_position, 
                radius=self.params.base_length * 0.1
            )
            
            # Combinaison des signaux locaux et globaux
            combined_activity = 0.7 * global_contraction + 0.3 * local_neural_input
            
            # Application des contraintes biomécaniques
            if i > 0:  # Propagation de la contraction depuis la base
                prev_contraction = self.segments[i-1].contraction
                combined_activity = 0.8 * combined_activity + 0.2 * prev_contraction
            
            # Mise à jour du segment
            segment.update(dt, combined_activity)
            
            # Mise à jour de la position de base du segment suivant
            if i < len(self.segments) - 1:
                next_segment = self.segments[i + 1]
                next_segment.base_x, next_segment.base_y = segment.tip_position
        
        # Mise à jour de l'état physiologique
        self._update_physiology(dt)
    
    def _process_stimuli(self, stimuli: Dict[str, float]):
        """Traite les stimuli externes affectant le tentacule"""
        for stimulus_type, intensity in stimuli.items():
            if stimulus_type == "mechanical":
                # Stimulation mécanique active les cnidocytes localement
                self._trigger_local_cnidocytes(intensity)
            elif stimulus_type == "chemical":
                # Stimuli chimiques augmentent la sensibilité globale
                self.sensitivity = min(2.0, self.sensitivity + intensity * 0.1)
                
    def _trigger_local_cnidocytes(self, intensity: float):
        """Déclenche les cnidocytes en réponse à un stimulus local"""
        discharged_count = 0
        
        for segment in self.segments:
            for cnidocyte in segment.cnidocytes:
                if cnidocyte.update(0.1, intensity * self.sensitivity):
                    discharged_count += 1
                    
                    # Stimulation locale du réseau nerveux
                    self.local_nerve_net.stimulate_region(
                        cnidocyte.x, 
                        cnidocyte.y,
                        radius=self.params.base_length * 0.15,
                        strength=0.5
                    )
        
        return discharged_count
    
    def _update_physiology(self, dt: float):
        """Met à jour l'état physiologique du tentacule"""
        # Consommation d'énergie basée sur l'activité
        total_contraction = sum(s.contraction for s in self.segments)
        energy_cost = 0.01 * dt * (1 + total_contraction)
        self.energy = max(0.0, self.energy - energy_cost)
        
        # Récupération progressive de la sensibilité
        self.sensitivity = max(1.0, self.sensitivity - 0.05 * dt)
        
        # Adaptation des cnidocytes basée sur l'énergie
        if self.energy < 0.3:
            for segment in self.segments:
                for cnidocyte in segment.cnidocytes:
                    cnidocyte.sensitivity = max(0.2, cnidocyte.sensitivity - 0.1 * dt)
    
    def get_prey_capture_points(self) -> List[Tuple[float, float]]:
        """
        Retourne les points de capture de proies (positions des cnidocytes actifs)
        """
        capture_points = []
        for segment in self.segments:
            for cnidocyte in segment.cnidocytes:
                if cnidocyte.is_charged:
                    capture_points.append((cnidocyte.x, cnidocyte.y))
        return capture_points

    def apply_force(self, force_x: float, force_y: float, point_x: float, point_y: float):
        """Applique une force externe à un point du tentacule"""
        # Trouve le segment le plus proche du point d'application
        closest_segment = min(self.segments, 
                            key=lambda s: ((s.base_x - point_x)**2 + 
                                         (s.base_y - point_y)**2)**0.5)
        
        # Calcul de l'angle de la force
        force_angle = np.arctan2(force_y, force_x)
        force_magnitude = (force_x**2 + force_y**2)**0.5
        
        # Application de la force au segment
        closest_segment.velocity += force_magnitude * 0.1
        closest_segment.target_angle += np.sign(force_angle - closest_segment.angle) * 0.1        