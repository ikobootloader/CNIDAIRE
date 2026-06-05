import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .nervous_system import DiffuseNerveNet
from .tentacles import Tentacle, TentacleParameters
from .digestive_system import GastricCavity, DigestiveState
from .feeding_system import FeedingSystem, PreyParameters
from .immune_system import ImmuneSystem, ImmuneParameters, ThreatType
from .growth_system import GrowthSystem, GrowthParameters
from .regeneration_system import RegenerationSystem
from .reproduction_system import ReproductionSystem
from src.config.settings import sim_config

@dataclass
class CnidarianConfig:
    """Configuration globale du cnidaire"""
    # Dimensions
    body_radius: float = 60.0  # Augmenté de 30.0 à 60.0
    tentacle_length: float = 80.0  # Augmenté de 40.0 à 80.0
    n_tentacles: int = 8
    
    # Paramètres physiologiques
    max_energy: float = 100.0
    basal_metabolism: float = 0.01
    regeneration_rate: float = 0.05
    
    # Paramètres comportementaux
    swimming_force: float = 2.0
    rotation_speed: float = 0.1
    max_speed: float = 5.0

class CnidarianOrganism:
    """
    Classe principale représentant un cnidaire complet avec tous ses systèmes
    """
    def __init__(self, x: float, y: float, config: Optional[CnidarianConfig] = None):
        self.config = config or CnidarianConfig()
        
        # Position et orientation
        self.x = x
        self.y = y
        self.orientation = 0.0  # en radians
        self.velocity = np.zeros(2)
        
        # Ajout des propriétés de taille de base
        self._base_radius = self.config.body_radius
        self.mass = 1.0  # masse unitaire par défaut        
        
        # Systèmes principaux
        self.nerve_net = DiffuseNerveNet(
            n_neurons=200,  # Plus de neurones pour le réseau global
            radius=self.config.body_radius
        )
        
        # Création des tentacules
        self.tentacles = self._initialize_tentacles()
        
        # Système digestif
        self.gastric_cavity = GastricCavity(
            center_x=self.x,
            center_y=self.y
        )
        
        # Système d'alimentation
        self.feeding_system = FeedingSystem(
            tentacles=self.tentacles,
            gastric_cavity=self.gastric_cavity,
            mouth_position=(self.x, self.y)
        )
        
        # État physiologique
        self.energy = self.config.max_energy
        self.health = 1.0
        self.age = 0.0
        
        # États comportementaux
        self.is_active = False      # Ajout de l'état d'activité
        self.is_swimming = False
        self.is_feeding = False
        self.is_regenerating = False
        self.global_activity = 0.0
        self.swim_phase = 0.0
        self.body_pulse = 0.0
        self.pulse_frequency = 0.0
        self.target_prey_position = None
        self.current_vector = np.zeros(2, dtype=float)
        self.swim_burst_timer = 0.0
        self.rest_phase_timer = 0.0

        # Systèmes principaux
        self.nerve_net = DiffuseNerveNet(
            n_neurons=200,  
            radius=self.config.body_radius
        )
        
        # Mémoire des stimuli
        self.stimulus_history = []
        
        # Nouveaux systèmes
        self.immune_system = ImmuneSystem()
        self.growth_system = GrowthSystem()
        self.regeneration_system = RegenerationSystem()
        self.reproduction_system = ReproductionSystem(self)
        self.growth_system.size = 50.0  # Taille initiale plus grande        
        
        # Mise à jour des états physiologiques pour inclure les nouveaux systèmes
        self.vitality = 1.0
        self.developmental_stage = None
        self.size_factor = 1.0  # Facteur d'échelle basé sur la croissance       
        self.radius = self._base_radius
        self.size = self.growth_system.size
        self.last_collision_force = 0.0
        self.collision_count = 0
        
    def _get_environmental_conditions(self, environment_state: Optional[Dict] = None) -> Dict:
        """
        Extrait et normalise les conditions environnementales pertinentes
        
        Args:
            environment_state: État de l'environnement fourni par la simulation
            
        Returns:
            Dict contenant les conditions environnementales normalisées
        """
        if environment_state is None:
            environment_state = {}
            
        # Valeurs par défaut pour les conditions essentielles
        conditions = {
            'temperature': environment_state.get('temperature', 20.0),
            'oxygen_level': environment_state.get('oxygen_level', environment_state.get('oxygen', 0.9)),
            'nutrients': environment_state.get('nutrients', 0.8),
            'light_intensity': environment_state.get('light_intensity', 1.0),
            'water_current': environment_state.get('water_current', environment_state.get('current', (0.0, 0.0))),
            'salinity': environment_state.get('salinity', 1.0),
            'ph': environment_state.get('ph', 7.0),
            'pressure': environment_state.get('pressure', 1.0)
        }
        
        # Application des limites physiologiques
        conditions['temperature'] = max(15.0, min(25.0, conditions['temperature']))
        conditions['oxygen_level'] = max(0.0, min(1.0, conditions['oxygen_level']))
        conditions['nutrients'] = max(0.0, min(1.0, conditions['nutrients']))
        conditions['light_intensity'] = max(0.0, min(1.0, conditions['light_intensity']))
        
        # Calcul de conditions dérivées
        conditions['stress_level'] = self._calculate_stress_level(conditions)
        
        return conditions

    def _calculate_stress_level(self, conditions: Dict) -> float:
        """
        Calcule le niveau de stress physiologique basé sur les conditions environnementales
        
        Args:
            conditions: Dictionnaire des conditions environnementales
            
        Returns:
            Niveau de stress entre 0.0 (minimal) et 1.0 (maximal)
        """
        stress_factors = [
            # Stress thermique
            abs(conditions['temperature'] - 20.0) / 5.0,  # Écart à la température optimale
            
            # Stress hypoxique
            (1.0 - conditions['oxygen_level']) * 1.5,
            
            # Stress nutritif
            (1.0 - conditions['nutrients']) * 0.8,
            
            # Stress osmotique
            abs(conditions['salinity'] - 1.0) * 1.2,
            
            # Stress pH
            abs(conditions['ph'] - 7.0) * 0.5
        ]
        
        # Moyenne pondérée des facteurs de stress
        total_stress = sum(max(0.0, min(1.0, factor)) for factor in stress_factors)
        return min(1.0, total_stress / len(stress_factors))
        
    def _initialize_tentacles(self) -> List[Tentacle]:
        """Initialise les tentacules autour du corps"""
        tentacles = []
        for i in range(self.config.n_tentacles):
            angle = (2 * np.pi * i) / self.config.n_tentacles + self.orientation
            base_x = self.x + self.config.body_radius * np.cos(angle)
            base_y = self.y + self.config.body_radius * np.sin(angle)
            tentacles.append(Tentacle(base_x, base_y, angle))
        return tentacles
        
    def update(self, dt: float, environment_state: Dict = None) -> List['CnidarianOrganism']:
        """Met à jour l'état global du cnidaire"""
        # Obtention des conditions environnementales
        conditions = self._get_environmental_conditions(environment_state)
        
        # Mise à jour des systèmes biologiques
        growth_status = self.growth_system.update(dt, {
            'energy_level': self.energy / self.config.max_energy,
            'temperature': conditions.get('temperature', 20.0),
            'nutrient_level': conditions.get('nutrients', 1.0)
        })
        
        # Mise à jour des paramètres basés sur la croissance
        self._update_size_dependent_parameters(growth_status)        
        
        # Mise à jour de l'état physiologique
        self._update_physiology(dt)
        
        # Traitement des stimuli environnementaux
        if environment_state:
            self._process_environment(environment_state)
        
        # Mise à jour du réseau nerveux central
        neural_response = self.nerve_net.update(dt)
        self.global_activity = float(np.mean(neural_response))

        immune_status = self.immune_system.update(
            dt,
            nerve_activity=self.global_activity
        )
        
        # Mise à jour des comportements basée sur l'activité neurale
        self._update_behaviors(neural_response, dt)

        # Mise à jour de l'état d'activité global
        self.is_active = (
            self.is_swimming or
            self.is_feeding or
            self.global_activity > 0.3 or
            np.linalg.norm(self.velocity) > 0.1
        )
        
        # Mise à jour des systèmes
        offspring = self._update_systems(dt, conditions)
        
        # Mise à jour de la position
        self._update_position(dt)
        
        # Mise à jour de l'âge
        self.age += dt
        
        # Mise à jour de la vitalité globale
        self._update_vitality(growth_status, immune_status)

        return offspring
        
    def _process_environment(self, environment_state: Dict):
        """Traite les informations environnementales"""
        # Détection des proies
        nearest_prey = None
        nearest_distance = float('inf')
        if 'prey' in environment_state:
            for prey in environment_state['prey']:
                prey_distance = np.sqrt(
                    (prey['position'][0] - self.x) ** 2 +
                    (prey['position'][1] - self.y) ** 2
                )
                if prey_distance < nearest_distance:
                    nearest_distance = prey_distance
                    nearest_prey = prey['position']
                if self._is_prey_in_range(prey['position']):
                    self.feeding_system.attempt_prey_capture(
                        prey['position'],
                        PreyParameters(**prey['parameters']),
                        prey.get('world_prey_id')
                    )
        self.target_prey_position = nearest_prey
                    
        # Détection des menaces
        if 'threats' in environment_state:
            for threat in environment_state['threats']:
                if self._is_threat_in_range(threat['position']):
                    self._trigger_defense_response(threat)
                    
        # Conditions environnementales
        if 'water_current' in environment_state:
            self.current_vector = np.array(environment_state['water_current'], dtype=float)
            self._adapt_to_current(environment_state['water_current'])
        else:
            self.current_vector = np.zeros(2, dtype=float)
            
    def _update_physiology(self, dt: float):
        """Met à jour l'état physiologique"""
        # Métabolisme de base
        energy_consumption = self.config.basal_metabolism * dt
        
        # Coût énergétique des activités
        if self.is_swimming:
            energy_consumption += 0.05 * dt * np.linalg.norm(self.velocity)
        if self.is_feeding:
            energy_consumption += 0.03 * dt
            
        # Consommation d'énergie
        self.energy = max(0, self.energy - energy_consumption)
        
        # Régénération si suffisamment d'énergie
        if self.energy > self.config.max_energy * 0.7 and self.health < 1.0:
            self.health = min(1.0, self.health + self.config.regeneration_rate * dt)
            self.is_regenerating = True
        else:
            self.is_regenerating = False
            
        # Mise à jour de la santé basée sur l'énergie
        if self.energy < self.config.max_energy * 0.2:
            self.health = max(0, self.health - 0.01 * dt)
    
    def _update_behaviors(self, neural_response: np.ndarray, dt: float):
        """Met à jour les comportements basés sur l'activité neurale"""
        # Analyse de l'activité neurale par région
        anterior_activity = np.mean(neural_response[:len(neural_response)//2])
        posterior_activity = np.mean(neural_response[len(neural_response)//2:])

        self._update_orientation_targets(anterior_activity, posterior_activity)
        
        # Décision de nage
        swim_threshold = 0.4 - (1.0 - self.energy/self.config.max_energy) * 0.2
        prey_drive = 0.18 if self.target_prey_position is not None else 0.0
        if anterior_activity + prey_drive > swim_threshold and self._can_start_swim_burst(dt):
            self._initiate_swimming(anterior_activity, dt)
        else:
            self._reduce_swimming(dt)
            
        # Coordination des tentacules pendant la nage
        if self.is_swimming:
            self._coordinate_swimming_tentacles(neural_response, dt)
            
        # Réponses aux stimuli mémorisés
        self._process_stimulus_memory()

    def _update_orientation_targets(self, anterior_activity: float, posterior_activity: float):
        """Oriente la nage vers une cible pertinente ou une dérive naturelle."""
        turn_rate = 0.4 + 0.8 * max(0.0, anterior_activity - posterior_activity)

        if self.target_prey_position is not None:
            target_angle = np.arctan2(
                self.target_prey_position[1] - self.y,
                self.target_prey_position[0] - self.x
            )
            self._rotate_towards(target_angle, rate=turn_rate)
            return

        current_strength = float(np.linalg.norm(self.current_vector))
        if current_strength > 0.05:
            drift_angle = np.arctan2(self.current_vector[1], self.current_vector[0])
            cruise_offset = np.pi * 0.35 * np.sin(self.age * 0.25 + self.x * 0.002)
            self._rotate_towards(drift_angle + cruise_offset, rate=0.25)
            return

        roaming_angle = self.orientation + np.sin(self.age * 0.3 + self.y * 0.002) * 0.08
        self._rotate_towards(roaming_angle, rate=0.12)

    def _can_start_swim_burst(self, dt: float) -> bool:
        """Alterne les phases d'effort et de relâchement pour éviter une nage uniforme."""
        if self.target_prey_position is not None:
            self.rest_phase_timer = 0.0
            self.swim_burst_timer = max(0.45, self.swim_burst_timer)

        if self.rest_phase_timer > 0.0:
            self.rest_phase_timer = max(0.0, self.rest_phase_timer - dt)
            self.swim_burst_timer = 0.0
            return False

        if self.swim_burst_timer <= 0.0:
            self.swim_burst_timer = 0.45 + 0.55 * min(1.0, self.energy / self.config.max_energy)

        self.swim_burst_timer = max(0.0, self.swim_burst_timer - dt)
        if self.swim_burst_timer == 0.0:
            self.rest_phase_timer = 0.18 + 0.22 * max(0.0, 1.0 - self.energy / self.config.max_energy)

        return True
        
    def _initiate_swimming(self, intensity: float, dt: float):
        """Initie ou maintient le mouvement de nage"""
        self.is_swimming = True
        self.pulse_frequency = 1.8 + intensity * 1.6
        self.swim_phase = (self.swim_phase + dt * self.pulse_frequency * 2 * np.pi) % (2 * np.pi)
        pulse_envelope = 0.5 + 0.5 * np.sin(self.swim_phase)
        propulsion_pulse = pulse_envelope ** 1.5
        self.body_pulse = 0.2 + 0.8 * propulsion_pulse

        # Calcul de la force de propulsion
        swim_force = self.config.swimming_force * intensity * (0.35 + propulsion_pulse)
        
        # Direction basée sur l'orientation
        force_direction = np.array([
            np.cos(self.orientation),
            np.sin(self.orientation)
        ])
        
        # Application de la force
        acceleration = force_direction * swim_force
        self.velocity += acceleration * dt
        
        # Limitation de la vitesse
        speed = np.linalg.norm(self.velocity)
        if speed > self.config.max_speed:
            self.velocity *= self.config.max_speed / speed
            
    def _reduce_swimming(self, dt: float):
        """Réduit progressivement la vitesse de nage"""
        self.is_swimming = False
        self.pulse_frequency = max(0.0, self.pulse_frequency - dt * 2.0)
        self.body_pulse = max(0.0, self.body_pulse - dt * 1.5)
        
        # Friction de l'eau
        friction = 0.95
        self.velocity *= friction ** dt

        if np.linalg.norm(self.current_vector) > 0.0:
            self.velocity += self.current_vector * (0.08 * dt)
        
        # Arrêt complet si vitesse très faible
        if np.linalg.norm(self.velocity) < 0.1:
            self.velocity = np.zeros(2)
            
    def _coordinate_swimming_tentacles(self, neural_response: np.ndarray, dt: float):
        """Coordonne les tentacules pendant la nage"""
        # Phase de la pulsation de nage
        swim_phase = self.swim_phase
        
        for i, tentacle in enumerate(self.tentacles):
            # Phase décalée pour chaque tentacle
            tentacle_phase = swim_phase + (2 * np.pi * i) / len(self.tentacles)
            
            # Mouvement ondulatoire
            base_angle = self.orientation + (i * 2 * np.pi) / len(self.tentacles)
            speed_factor = min(1.0, np.linalg.norm(self.velocity) / max(1e-6, self.config.max_speed))
            wave_amplitude = 0.18 + 0.22 * self.body_pulse + 0.12 * speed_factor
            
            for j, segment in enumerate(tentacle.segments):
                # Propagation de l'onde le long du tentacule
                segment_phase = tentacle_phase - j * 0.2
                segment.target_angle = (
                    base_angle + 
                    wave_amplitude * np.sin(segment_phase) * 
                    (1.0 - j/len(tentacle.segments))
                )
                
    def _update_position(self, dt: float):
        """Met à jour la position du cnidaire"""
        # Déplacement
        self.x += self.velocity[0] * dt
        self.y += self.velocity[1] * dt
        
        # Mise à jour des positions des sous-systèmes
        self._update_subsystem_positions()
        
    def _update_subsystem_positions(self):
        """Met à jour les positions de tous les sous-systèmes"""
        # Mise à jour de la cavité gastrique
        self.gastric_cavity.center_x = self.x
        self.gastric_cavity.center_y = self.y
        
        # Mise à jour des bases des tentacules
        for i, tentacle in enumerate(self.tentacles):
            angle = self.orientation + (2 * np.pi * i) / len(self.tentacles)
            tentacle.base_x = self.x + self.radius * np.cos(angle)
            tentacle.base_y = self.y + self.radius * np.sin(angle)
            tentacle.base_angle = angle
            
        # Mise à jour du système d'alimentation
        self.feeding_system.mouth_position = (self.x, self.y)
        
    def _trigger_defense_response(self, threat: Dict):
        """Déclenche une réponse défensive face à une menace"""
        threat_position = np.array(threat['position'])
        threat_direction = threat_position - np.array([self.x, self.y])
        threat_distance = np.linalg.norm(threat_direction)
        
        if threat_distance < self.config.body_radius * 3:
            # Contraction rapide
            self._contract_body()
            
            # Réorientation pour l'évasion
            escape_direction = -threat_direction / threat_distance
            target_orientation = np.arctan2(escape_direction[1], escape_direction[0])
            self._rotate_towards(target_orientation, rate=2.0)
            
            # Stimulation du réseau nerveux pour la fuite
            self.nerve_net.stimulate_region(
                self.x, self.y,
                radius=self.config.body_radius,
                strength=1.0
            )

    def _rotate_towards(self, target_angle: float, rate: float = 1.0):
        """Rotation progressive vers un angle cible"""
        angle_diff = (target_angle - self.orientation + np.pi) % (2 * np.pi) - np.pi
        self.orientation += np.sign(angle_diff) * min(
            abs(angle_diff),
            self.config.rotation_speed * rate
        )
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

    def _is_prey_in_range(self, prey_position: Tuple[float, float]) -> bool:
        """Détermine si une proie est à portée de détection"""
        # Rayon de détection basé sur la longueur des tentacules
        detection_radius = max(tentacle.params.base_length for tentacle in self.tentacles)
        detection_radius *= 1.2  # Marge supplémentaire
        
        # Position relative de la proie
        dx = prey_position[0] - self.x
        dy = prey_position[1] - self.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Vérification de la distance
        if distance > detection_radius:
            return False
            
        # Vérification si la proie est dans le champ des tentacules
        for tentacle in self.tentacles:
            if self._is_within_tentacle_range(prey_position, tentacle):
                return True
                
        return False
        
    def _is_within_tentacle_range(self, position: Tuple[float, float], tentacle: Tentacle) -> bool:
        """Vérifie si une position est à portée d'un tentacule spécifique"""
        # Angle vers la position
        dx = position[0] - tentacle.base_x
        dy = position[1] - tentacle.base_y
        angle_to_position = np.arctan2(dy, dx)
        
        # Angle normalisé par rapport à l'orientation du tentacule
        relative_angle = (angle_to_position - tentacle.base_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Distance à la base du tentacule
        distance = np.sqrt(dx**2 + dy**2)
        
        # Vérification des conditions
        return (abs(relative_angle) < np.pi/3 and  # Dans l'arc du tentacule
                distance < tentacle.params.base_length)  # À portée
                
    def _is_threat_in_range(self, threat_position: Tuple[float, float]) -> bool:
        """Détermine si une menace est suffisamment proche pour être détectée"""
        threat_detection_radius = self.config.body_radius * 3
        
        dx = threat_position[0] - self.x
        dy = threat_position[1] - self.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # La sensibilité aux menaces augmente quand l'énergie est faible
        sensitivity_factor = 1.0 + max(0, 0.5 * (1.0 - self.energy/self.config.max_energy))
        
        return distance < threat_detection_radius * sensitivity_factor
        
    def _adapt_to_current(self, current_vector: Tuple[float, float]):
        """Adapte le comportement en fonction du courant d'eau"""
        current_strength = np.sqrt(current_vector[0]**2 + current_vector[1]**2)
        current_direction = np.arctan2(current_vector[1], current_vector[0])
        
        if current_strength > self.config.max_speed * 0.5:
            # Réorientation pour minimiser la résistance
            if self.energy > self.config.max_energy * 0.3:
                # Face au courant si assez d'énergie
                target_orientation = current_direction + np.pi
            else:
                # Dans le sens du courant si faible énergie
                target_orientation = current_direction
                
            self._rotate_towards(target_orientation, rate=0.5)
            
            # Ajustement des tentacules
            self._adapt_tentacles_to_current(current_vector)
            
    def _adapt_tentacles_to_current(self, current_vector: Tuple[float, float]):
        """Ajuste la position des tentacules en fonction du courant"""
        current_angle = np.arctan2(current_vector[1], current_vector[0])
        current_strength = np.sqrt(current_vector[0]**2 + current_vector[1]**2)
        
        for tentacle in self.tentacles:
            # Calcul de l'angle relatif au courant
            relative_angle = (tentacle.base_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
            
            # Facteur de flexion basé sur l'angle relatif
            bending_factor = np.sin(relative_angle) * current_strength * 0.1
            
            # Application de la flexion aux segments
            for i, segment in enumerate(tentacle.segments):
                segment_factor = bending_factor * (i + 1) / len(tentacle.segments)
                segment.target_angle += segment_factor
                
    def _contract_body(self):
        """Contraction défensive rapide du corps"""
        # Réduction temporaire du rayon
        contraction_factor = 0.7
        
        # Contraction des tentacules
        for tentacle in self.tentacles:
            for segment in tentacle.segments:
                segment.contraction = 0.8  # Forte contraction
                
        # Stimulation du système nerveux
        self.nerve_net.stimulate_region(
            self.x, self.y,
            radius=self.config.body_radius * contraction_factor,
            strength=1.0
        )
        
        # Consommation d'énergie pour la contraction
        self.energy = max(0, self.energy - 5.0)
        
    def _process_stimulus_memory(self):
        """Traite la mémoire des stimuli récents"""
        if not self.stimulus_history:
            return
            
        # Analyse des patterns de stimuli récents
        recent_stimuli = self.stimulus_history[-10:]
        
        # Détection de patterns répétitifs
        if self._detect_stimulus_pattern(recent_stimuli):
            self._adapt_to_pattern(recent_stimuli)
            
        # Nettoyage de la mémoire ancienne
        while len(self.stimulus_history) > 100:  # Limite de mémoire
            self.stimulus_history.pop(0)
            
    def _detect_stimulus_pattern(self, stimuli: List[Dict]) -> bool:
        """Détecte des patterns dans les stimuli récents"""
        if len(stimuli) < 4:
            return False
            
        # Extraction des types de stimuli
        stimulus_types = [s['type'] for s in stimuli]
        
        # Recherche de séquences répétées
        for pattern_length in range(2, len(stimulus_types)//2 + 1):
            pattern = stimulus_types[-pattern_length:]
            if pattern in stimulus_types[:-pattern_length]:
                return True
                
        return False
        
    def _adapt_to_pattern(self, stimuli: List[Dict]):
        """Adapte le comportement en fonction des patterns détectés"""
        # Types de stimuli dominants
        stimulus_counts = {}
        for stimulus in stimuli:
            stimulus_type = stimulus['type']
            stimulus_counts[stimulus_type] = stimulus_counts.get(stimulus_type, 0) + 1
            
        dominant_stimulus = max(stimulus_counts.items(), key=lambda x: x[1])[0]
        
        # Adaptation selon le type dominant
        if dominant_stimulus == 'prey':
            # Augmentation de la sensibilité des tentacules
            for tentacle in self.tentacles:
                for segment in tentacle.segments:
                    for cnidocyte in segment.cnidocytes:
                        cnidocyte.sensitivity *= 1.2
                        
        elif dominant_stimulus == 'threat':
            # Augmentation de la réactivité défensive
            self.config.rotation_speed *= 1.1
            self.config.max_speed *= 1.1
            
    def _update_size_dependent_parameters(self, growth_status: Dict):
        """Met à jour les paramètres qui dépendent de la taille"""
        # Mise à jour du facteur de taille
        new_size = growth_status['size']
        self.size = new_size
        self.size_factor = new_size / self.config.body_radius
        
        # Mise à jour du rayon du corps
        self.radius = self.config.body_radius * self.size_factor
        
        # Ajustement des paramètres des tentacules
        for tentacle in self.tentacles:
            tentacle.params.base_length = (
                self.config.tentacle_length * self.size_factor
            )
            # Recalcul des positions des segments
            self._adjust_tentacle_segments(tentacle)
            
        # Ajustement de la cavité gastrique
        self.gastric_cavity.params.cavity_radius = (
            self.radius * 0.6  # 60% du rayon du corps
        )
        
    def _adjust_tentacle_segments(self, tentacle: Tentacle):
        """Ajuste les segments des tentacules après changement de taille"""
        # Recalcul de la longueur des segments
        new_segment_length = (
            tentacle.params.base_length / len(tentacle.segments)
        )
        
        # Mise à jour de chaque segment
        current_x = tentacle.base_x
        current_y = tentacle.base_y
        current_angle = tentacle.base_angle
        
        for segment in tentacle.segments:
            segment.length = new_segment_length
            segment.base_x = current_x
            segment.base_y = current_y
            
            # Calcul de la nouvelle position de l'extrémité
            tip_x = current_x + segment.length * np.cos(current_angle)
            tip_y = current_y + segment.length * np.sin(current_angle)
            
            # Mise à jour pour le segment suivant
            current_x, current_y = tip_x, tip_y
            
    def _update_vitality(self, growth_status: Dict, immune_status: Dict):
        """Met à jour l'état de vitalité global de l'organisme"""
        # Facteurs contribuant à la vitalité
        vitality_factors = {
            'growth': growth_status['vitality'],
            'immune': 1.0 - immune_status.get('global_activation', 0),
            'energy': self.energy / self.config.max_energy,
            'health': self.health
        }
        
        # Calcul de la vitalité pondérée
        weights = {
            'growth': 0.3,
            'immune': 0.2,
            'energy': 0.25,
            'health': 0.25
        }
        
        self.vitality = sum(
            factor * weights[name]
            for name, factor in vitality_factors.items()
        )
        
        # Mise à jour du stade de développement
        self.developmental_stage = growth_status['stage']

    def set_size(self, new_size: float):
        """Met à jour la taille de l'organisme et ses dépendances morphologiques."""
        self.growth_system.size = new_size
        self._update_size_dependent_parameters({'size': new_size})

    def handle_threat(self, threat_type: str, position: Tuple[float, float], 
                     intensity: float) -> bool:
        """
        Gère une menace détectée (pathogène, toxine, dommage physique)
        Retourne True si la menace a été traitée avec succès
        """
        # Conversion en type de menace immunitaire
        threat_mapping = {
            'pathogen': ThreatType.PATHOGEN,
            'toxin': ThreatType.TOXIN,
            'physical': ThreatType.PHYSICAL_DAMAGE,
            'foreign': ThreatType.FOREIGN_TISSUE
        }
        immune_threat = threat_mapping.get(threat_type)
        
        if not immune_threat:
            return False
            
        # Vérification de la distance par rapport au corps
        distance = np.sqrt(
            (position[0] - self.x)**2 + 
            (position[1] - self.y)**2
        )
        
        if distance > self.radius * 1.5:  # Menace trop éloignée
            return False
            
        # Tentative de réponse immunitaire
        immune_response = self.immune_system.detect_threat(
            immune_threat,
            position,
            intensity
        )
        
        if immune_response:
            # Stimulation du réseau nerveux
            self.nerve_net.stimulate_region(
                position[0],
                position[1],
                radius=10.0,
                strength=intensity * 0.5
            )
            
            # Réponse comportementale
            self._trigger_defense_behavior(threat_type, position, intensity)
            
        return immune_response
        
    def _trigger_defense_behavior(self, threat_type: str, 
                                position: Tuple[float, float],
                                intensity: float):
        """Déclenche une réponse comportementale à une menace"""
        # Calcul de la direction de la menace
        threat_direction = np.array([
            position[0] - self.x,
            position[1] - self.y
        ], dtype=float)
        threat_distance = np.linalg.norm(threat_direction)
        
        if threat_distance > 0:
            threat_direction /= threat_distance
            
        # Réponses comportementales selon le type de menace
        if threat_type == 'physical':
            # Contraction rapide
            self._contract_body()
            
            # Fuite dans la direction opposée
            escape_force = intensity * self.config.swimming_force * 2.0
            self.velocity -= threat_direction * escape_force
            
        elif threat_type in ['pathogen', 'toxin']:
            # Augmentation de la production de mucus (simulation)
            self.health = max(0, self.health - intensity * 0.1)
            
            # Tentative d'expulsion
            self._trigger_expulsion(position)
            
        elif threat_type == 'foreign':
            # Intensification de l'activité des cnidocytes
            self._activate_nearby_cnidocytes(position)
            
    def _contract_body(self):
        """Contraction défensive rapide du corps"""
        # Facteurs de contraction
        contraction_factor = 0.7  # Réduction à 70% de la taille normale
        contraction_duration = 2.0  # secondes
        
        # Réduction temporaire du rayon
        self.contracting = True
        self._previous_radius = self.radius
        self.radius *= contraction_factor
        
        # Contraction des tentacules
        for tentacle in self.tentacles:
            for segment in tentacle.segments:
                segment.contraction = 0.8  # Forte contraction
                
        # Stimulation du réseau nerveux
        self.nerve_net.stimulate_region(
            self.x, self.y,
            radius=self.radius,
            strength=1.0
        )
        
        # Consommation d'énergie pour la contraction
        energy_cost = 5.0
        self.energy = max(0, self.energy - energy_cost)
        
    def _trigger_expulsion(self, position: Tuple[float, float]):
        """Déclenche une expulsion locale de mucus et d'eau"""
        # Création d'une onde de pression
        self.gastric_cavity.current_state = DigestiveState.EXPELLING
        expulsion_force = 2.0
        
        # Application de la force aux tentacules proches
        for tentacle in self.tentacles:
            # Distance à la zone d'expulsion
            distance = np.sqrt(
                (tentacle.base_x - position[0])**2 +
                (tentacle.base_y - position[1])**2
            )
            
            if distance < self.radius:
                # Force d'expulsion diminuant avec la distance
                force_magnitude = expulsion_force * (1 - distance/self.radius)
                direction = np.array([
                    tentacle.base_x - position[0],
                    tentacle.base_y - position[1]
                ])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    
                # Application de la force
                tentacle.apply_force(
                    force_magnitude * direction[0],
                    force_magnitude * direction[1],
                    position[0], position[1]
                )
                
    def _activate_nearby_cnidocytes(self, position: Tuple[float, float]):
        """Active les cnidocytes autour d'une position donnée"""
        activation_radius = self.radius * 0.3
        activated_count = 0
        
        for tentacle in self.tentacles:
            for segment in tentacle.segments:
                for cnidocyte in segment.cnidocytes:
                    # Distance au point d'activation
                    distance = np.sqrt(
                        (cnidocyte.x - position[0])**2 +
                        (cnidocyte.y - position[1])**2
                    )
                    
                    if (distance < activation_radius and 
                        cnidocyte.is_charged):
                        # Décharge du cnidocyte
                        cnidocyte.update(0.1, 1.0)  # Force le déclenchement
                        activated_count += 1
                        
                        # Stimulation locale du réseau nerveux
                        tentacle.local_nerve_net.stimulate_region(
                            cnidocyte.x,
                            cnidocyte.y,
                            radius=5.0,
                            strength=0.8
                        )
                        
        return activated_count
        
    def _update_systems(self, dt: float, conditions: Dict) -> List['CnidarianOrganism']:
        """
        Met à jour tous les sous-systèmes biologiques de l'organisme
        
        Args:
            dt: Pas de temps pour la mise à jour
        """
        # Mise à jour des tentacules
        contraction_signal = self.body_pulse if self.is_swimming else 0.0
        for tentacle in self.tentacles:
            tentacle.update(dt, contraction_signal)
            
        # Mise à jour du système digestif
        if self.gastric_cavity:
            self.gastric_cavity.update(dt)
            
        # Si le système d'alimentation est actif
        if self.feeding_system:
            self.feeding_system.update(dt)
            
        # Mise à jour de la régénération si présente
        if hasattr(self, 'regeneration_system') and self.regeneration_system:
            self.regeneration_system.update(
                dt,
                energy_available=self.energy,
                environmental_conditions=conditions
            )
            
        # Mise à jour de la reproduction si présente
        offspring_entities = []
        if hasattr(self, 'reproduction_system') and self.reproduction_system:
            offspring = self.reproduction_system.update(dt)
            if offspring:
                offspring_entities.append(offspring)
                
        # Mise à jour des attributs énergétiques
        self._update_energy(dt)

        return offspring_entities
        
    def _update_energy(self, dt: float):
        """
        Met à jour l'état énergétique de l'organisme
        
        Args:
            dt: Pas de temps pour la mise à jour
        """
        # Coût énergétique de base
        base_cost = self.config.basal_metabolism * dt
        
        # Coûts additionnels basés sur l'activité
        activity_costs = {
            'swimming': 0.05 * dt * float(self.is_swimming),
            'feeding': 0.03 * dt * float(self.is_feeding),
            'neural': 0.02 * dt * self.global_activity,
            'regeneration': 0.04 * dt * float(self.is_regenerating)
        }
        
        total_cost = base_cost + sum(activity_costs.values())
        
        # Application des coûts énergétiques
        self.energy = max(0.0, self.energy - total_cost)
        
        # Mise à jour de la santé basée sur l'énergie
        if self.energy < self.config.max_energy * 0.2:
            self.health = max(0.0, self.health - 0.01 * dt)
        elif self.energy > self.config.max_energy * 0.7:
            # Régénération lente si beaucoup d'énergie
            self.health = min(1.0, self.health + 0.005 * dt)

    def on_collision(self, other: 'CnidarianOrganism', impact_force: float):
        """Réagit à une collision physique avec un autre organisme."""
        self.last_collision_force = impact_force
        self.collision_count += 1
        self.health = max(0.0, self.health - impact_force * 0.001)
        self.nerve_net.stimulate_region(
            self.x,
            self.y,
            radius=max(5.0, self.radius * 0.25),
            strength=min(1.0, impact_force * 0.1)
        )
