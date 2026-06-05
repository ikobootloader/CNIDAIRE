import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class NeuronParameters:
    """Paramètres biologiques des neurones de cnidaires"""
    # Potentiels membranaires (en mV)
    resting_potential: float = -70.0    # Potentiel de repos
    threshold: float = -55.0            # Seuil de déclenchement
    reset_potential: float = -75.0      # Potentiel post-décharge
    
    # Constantes temporelles (en ms)
    tau_membrane: float = 10.0          # Constante de temps membranaire
    tau_refactory: float = 2.0          # Période réfractaire
    tau_adaptation: float = 100.0       # Adaptation de l'excitabilité
    
    # Paramètres synaptiques
    synaptic_strength: float = 0.5      # Force des connexions
    plasticity_rate: float = 0.01       # Taux d'apprentissage
    max_connections: int = 15           # Nombre max de connexions par neurone

class CnidarianNeuron:
    """
    Modèle de neurone spécifique aux cnidaires, avec:
    - Signalisation bidirectionnelle
    - Pas de polarité axone/dendrite définie
    - Plasticité synaptique simple
    """
    def __init__(self, x: float, y: float, params: Optional[NeuronParameters] = None):
        self.x = x
        self.y = y
        self.params = params or NeuronParameters()
        
        # État membranaire
        self.membrane_potential = self.params.resting_potential
        self.refactory_period = 0.0
        self.adaptation_level = 0.0
        
        # Connexions synaptiques (bidirectionnelles)
        self.connections: List[Tuple[CnidarianNeuron, float]] = []  # (neurone, poids)
        self.synaptic_activity = 0.0
        
        # État d'activité
        self.is_active = False
        self.activity_history = []
        
    def connect_to(self, other: 'CnidarianNeuron', initial_weight: Optional[float] = None):
        """Établit une connexion bidirectionnelle avec un autre neurone"""
        if len(self.connections) >= self.params.max_connections:
            return
            
        # Poids initial basé sur la distance (décroissance exponentielle)
        if initial_weight is None:
            distance = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
            initial_weight = np.exp(-distance / 50.0) * self.params.synaptic_strength
        
        # Connexion bidirectionnelle
        self.connections.append((other, initial_weight))
        other.connections.append((self, initial_weight))
        
    def update(self, dt: float) -> float:
        """Mise à jour de l'état du neurone"""
        # Mise à jour de la période réfractaire
        if self.refactory_period > 0:
            self.refactory_period -= dt
            return 0.0
        
        # Intégration des entrées synaptiques
        synaptic_input = self._compute_synaptic_input()
        
        # Évolution du potentiel membranaire
        self.membrane_potential += dt * (
            -(self.membrane_potential - self.params.resting_potential) +
            synaptic_input - self.adaptation_level
        ) / self.params.tau_membrane
        
        # Déclenchement d'un potentiel d'action
        output = 0.0
        if self.membrane_potential >= self.params.threshold:
            output = 1.0
            self.membrane_potential = self.params.reset_potential
            self.refactory_period = self.params.tau_refactory
            self.adaptation_level += 0.1  # Adaptation à l'activité
            self.is_active = True
        else:
            self.is_active = False
            
        # Récupération de l'adaptation
        self.adaptation_level -= dt * self.adaptation_level / self.params.tau_adaptation
        
        # Mise à jour de l'historique
        self.activity_history.append(output)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
            
        return output
        
    def _compute_synaptic_input(self) -> float:
        """Calcule la somme des entrées synaptiques"""
        total_input = 0.0
        for neuron, weight in self.connections:
            if neuron.is_active:
                total_input += weight
                
        return total_input * self.params.synaptic_strength

class DiffuseNerveNet:
    """
    Réseau nerveux diffus caractéristique des cnidaires:
    - Organisation non hiérarchique
    - Propagation multidirectionnelle
    - Intégration locale
    """
    def __init__(self, n_neurons: int, radius: float):
        self.radius = radius
        
        # Création des neurones avec distribution spatiale
        self.neurons: List[CnidarianNeuron] = []
        for _ in range(n_neurons):
            # Position aléatoire dans un anneau (concentration près du bord)
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.normal(0.8*radius, 0.1*radius)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            
            self.neurons.append(CnidarianNeuron(x, y))
            
        # Établissement des connexions locales
        self._create_local_connectivity()
        
        # Métriques du réseau
        self.global_activity = 0.0
        self.activity_patterns: List[np.ndarray] = []
        
    def _create_local_connectivity(self):
        """Crée des connexions entre neurones proches"""
        for i, neuron in enumerate(self.neurons):
            for other in self.neurons[i+1:]:
                # Probabilité de connexion décroissante avec la distance
                distance = np.sqrt(
                    (neuron.x - other.x)**2 + 
                    (neuron.y - other.y)**2
                )
                
                if distance < self.radius * 0.3:  # Connexion locale
                    connection_prob = np.exp(-distance / (self.radius * 0.2))
                    if np.random.random() < connection_prob:
                        neuron.connect_to(other)
    
    def update(self, dt: float) -> np.ndarray:
        """Met à jour l'état du réseau nerveux et retourne l'activité globale"""
        # Collecte des activités neuronales
        activities = np.zeros(len(self.neurons))
        
        for i, neuron in enumerate(self.neurons):
            activities[i] = neuron.update(dt)
            
        # Mise à jour de l'activité globale avec décroissance temporelle
        self.global_activity = 0.8 * self.global_activity + 0.2 * np.mean(activities)
        
        # Stockage du motif d'activité
        self.activity_patterns.append(activities)
        if len(self.activity_patterns) > 100:  # Limite de mémoire
            self.activity_patterns.pop(0)
            
        return activities
    
    def detect_activity_pattern(self) -> Dict[str, float]:
        """Analyse les motifs d'activité pour détecter des comportements"""
        if len(self.activity_patterns) < 10:
            return {"type": "random", "strength": 0.0}
            
        recent_patterns = np.array(self.activity_patterns[-10:])
        
        # Détection de motifs rythmiques (contraction de la méduse)
        fft_result = np.abs(np.fft.fft(recent_patterns.mean(axis=1)))
        main_freq = np.argmax(fft_result[1:len(fft_result)//2]) + 1
        rhythm_strength = fft_result[main_freq] / fft_result.sum()
        
        # Détection de vagues d'activité (propagation nerveuse)
        spatial_correlation = np.corrcoef(recent_patterns.T)
        wave_strength = np.mean(np.abs(spatial_correlation - np.eye(len(spatial_correlation))))
        
        # Classification du motif
        if rhythm_strength > 0.3:
            return {
                "type": "swimming",
                "strength": rhythm_strength,
                "frequency": main_freq
            }
        elif wave_strength > 0.4:
            return {
                "type": "feeding",
                "strength": wave_strength
            }
        else:
            return {
                "type": "random",
                "strength": max(rhythm_strength, wave_strength)
            }
            
    def get_local_activity(self, x: float, y: float, radius: float) -> float:
        """Calcule l'activité moyenne dans une région locale"""
        total_activity = 0.0
        count = 0
        
        for neuron in self.neurons:
            distance = np.sqrt((x - neuron.x)**2 + (y - neuron.y)**2)
            if distance <= radius:
                total_activity += neuron.is_active
                count += 1
                
        return total_activity / max(1, count)
    
    def stimulate_region(self, x: float, y: float, radius: float, strength: float):
        """Stimule les neurones dans une région donnée"""
        for neuron in self.neurons:
            distance = np.sqrt((x - neuron.x)**2 + (y - neuron.y)**2)
            if distance <= radius:
                # Injection de courant proportionnelle à la distance
                current = strength * (1 - distance/radius)
                neuron.membrane_potential += current

class BehavioralController:
    """
    Contrôleur des comportements de base du cnidaire:
    - Nage (contraction rythmique)
    - Alimentation
    - Réponse aux stimuli
    """
    def __init__(self, nerve_net: DiffuseNerveNet):
        self.nerve_net = nerve_net
        
        # États comportementaux
        self.swimming_state = 0.0  # Force de nage
        self.feeding_state = 0.0   # État d'alimentation
        self.defense_state = 0.0   # État défensif
        
        # Paramètres comportementaux
        self.swimming_threshold = 0.3
        self.feeding_threshold = 0.4
        self.defense_threshold = 0.5
        
        # Mémoire comportementale
        self.behavior_history = []
        self.last_behavior = "idle"
        
    def update(self, dt: float, stimuli: Dict[str, float] = None) -> str:
        """
        Mise à jour de l'état comportemental basée sur l'activité neurale
        et les stimuli externes
        """
        # Traitement des stimuli
        if stimuli:
            self._process_stimuli(stimuli)
            
        # Analyse du motif d'activité
        pattern = self.nerve_net.detect_activity_pattern()
        
        # Mise à jour des états comportementaux
        if pattern["type"] == "swimming":
            self.swimming_state = min(1.0, self.swimming_state + 0.1 * pattern["strength"])
            self.feeding_state *= 0.9
        elif pattern["type"] == "feeding":
            self.feeding_state = min(1.0, self.feeding_state + 0.1 * pattern["strength"])
            self.swimming_state *= 0.9
            
        # Sélection du comportement dominant
        behavior = self._select_behavior()
        
        # Mise à jour de l'historique
        self.behavior_history.append(behavior)
        if len(self.behavior_history) > 100:
            self.behavior_history.pop(0)
            
        self.last_behavior = behavior
        return behavior