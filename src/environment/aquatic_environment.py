import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
from src.config.settings import sim_config

@dataclass
class EnvironmentConfig:
    """Configuration de l'environnement aquatique"""
    # Dimensions
    width: float = 1200.0
    height: float = 800.0
    
    # Paramètres physiques
    water_density: float = 1.0
    viscosity: float = 0.001
    temperature: float = 20.0
    
    # Paramètres du courant
    max_current_speed: float = 2.0
    current_variability: float = 0.3
    current_change_rate: float = 0.01
    
    # Paramètres chimiques
    oxygen_diffusion_rate: float = 0.1
    nutrient_diffusion_rate: float = 0.05
    
    # Paramètres de la grille de simulation
    grid_size: int = 50  # Nombre de cellules par côté
    
    # Paramètres environnementaux
    day_length: float = 300.0  # Durée du jour en secondes
    light_intensity_range: Tuple[float, float] = (0.2, 1.0)
    seasonal_cycle_length: float = 1200.0  # Durée d'une saison en secondes

class ZoneType(Enum):
    """Types de zones dans l'environnement"""
    OPEN_WATER = "open_water"
    SHALLOW = "shallow"
    DEEP = "deep"
    CURRENT = "current"
    NUTRIENT_RICH = "nutrient_rich"

class Environment:
    """Simulation de l'environnement aquatique"""
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        
        # Grilles de simulation
        self.grid_width = self.config.width / self.config.grid_size
        self.grid_height = self.config.height / self.config.grid_size
        
        # Initialisation des grilles
        self.current_grid = self._initialize_current_grid()
        self.temperature_grid = self._initialize_temperature_grid()
        self.oxygen_grid = self._initialize_oxygen_grid()
        self.nutrient_grid = self._initialize_nutrient_grid()
        
        # État environnemental
        self.time = 0.0
        self.day_phase = 0.0
        self.season_phase = 0.0
        
        # Zones environnementales
        self.zones = self._initialize_zones()
        
        # Cache pour les calculs
        self.flow_field_cache = {}
        self.last_update_time = time.time()
        
    def _initialize_current_grid(self) -> np.ndarray:
        """Initialise la grille des courants"""
        # Grille 3D : (x, y, [vélocité_x, vélocité_y])
        grid = np.zeros((self.config.grid_size, 
                        self.config.grid_size, 2))
        
        # Génération de courants de base
        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                # Courant de base avec variation sinusoïdale
                base_current = np.sin(j * 0.1) * self.config.max_current_speed * 0.5
                grid[i, j] = [base_current, 
                             np.cos(i * 0.1) * self.config.max_current_speed * 0.3]
                
        return grid
        
    def _initialize_temperature_grid(self) -> np.ndarray:
        """Initialise la grille de température"""
        grid = np.full((self.config.grid_size, self.config.grid_size),
                      self.config.temperature)
        
        # Gradient de température vertical (plus froid en profondeur)
        depth_gradient = np.linspace(0, -5, self.config.grid_size)
        grid += depth_gradient[:, np.newaxis]
        
        return grid
        
    def _initialize_oxygen_grid(self) -> np.ndarray:
        """Initialise la grille d'oxygène"""
        grid = np.ones((self.config.grid_size, self.config.grid_size))
        
        # Plus d'oxygène près de la surface
        depth_factor = np.linspace(1.2, 0.8, self.config.grid_size)
        grid *= depth_factor[:, np.newaxis]
        
        return grid
        
    def _initialize_nutrient_grid(self) -> np.ndarray:
        """Initialise la grille de nutriments"""
        grid = np.ones((self.config.grid_size, self.config.grid_size))
        
        # Plus de nutriments en profondeur
        depth_factor = np.linspace(0.8, 1.2, self.config.grid_size)
        grid *= depth_factor[:, np.newaxis]
        
        return grid
        
    def _initialize_zones(self) -> Dict[ZoneType, List[Tuple[int, int]]]:
        """Initialise les différentes zones de l'environnement"""
        zones = {zone_type: [] for zone_type in ZoneType}
        
        # Définition des zones
        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                depth = j / self.config.grid_size
                
                if depth < 0.3:
                    zones[ZoneType.SHALLOW].append((i, j))
                elif depth > 0.7:
                    zones[ZoneType.DEEP].append((i, j))
                else:
                    zones[ZoneType.OPEN_WATER].append((i, j))
                    
                # Zones de courant
                if 0.4 < depth < 0.6:
                    zones[ZoneType.CURRENT].append((i, j))
                    
                # Zones riches en nutriments (près du fond)
                if depth > 0.8:
                    zones[ZoneType.NUTRIENT_RICH].append((i, j))
                    
        return zones
    
    def update(self, dt: float):
        """Met à jour l'état de l'environnement"""
        self.time += dt
        
        # Mise à jour des cycles
        self._update_cycles(dt)
        
        # Mise à jour des grilles
        self._update_currents(dt)
        self._update_temperature(dt)
        self._update_oxygen(dt)
        self._update_nutrients(dt)
        
        # Mise à jour du cache
        self.last_update_time = time.time()
        self.flow_field_cache.clear()
        
    def _update_cycles(self, dt: float):
        """Met à jour les cycles jour/nuit et saisonniers"""
        # Cycle jour/nuit
        self.day_phase = (self.time % self.config.day_length) / self.config.day_length
        
        # Cycle saisonnier
        self.season_phase = (self.time % self.config.seasonal_cycle_length) / \
                           self.config.seasonal_cycle_length
                           
        # Influence sur les paramètres environnementaux
        self._apply_cycle_effects(dt)
        
    def _apply_cycle_effects(self, dt: float):
        """Applique les effets des cycles sur l'environnement"""
        # Intensité lumineuse basée sur le cycle jour/nuit
        light_intensity = self._calculate_light_intensity()
        
        # Effets saisonniers
        seasonal_temp_offset = np.sin(self.season_phase * 2 * np.pi) * 5.0
        
        # Application des effets
        self.temperature_grid += seasonal_temp_offset * dt
        self.oxygen_grid *= 1.0 + (light_intensity - 0.5) * 0.1 * dt
        
    def _update_currents(self, dt: float):
        """Met à jour les courants d'eau"""
        # Variation naturelle des courants
        noise = np.random.normal(
            0, 
            self.config.current_variability, 
            self.current_grid.shape
        )
        
        # Application progressive des variations
        self.current_grid += noise * dt * self.config.current_change_rate
        
        # Application des contraintes physiques
        self.current_grid = np.clip(
            self.current_grid,
            -self.config.max_current_speed,
            self.config.max_current_speed
        )
        
        # Conservation de la masse (divergence nulle)
        self._enforce_incompressibility()
        
    def _enforce_incompressibility(self):
        """Assure la conservation de la masse dans les courants"""
        # Calcul de la divergence
        div_x = np.gradient(self.current_grid[:,:,0], axis=0)
        div_y = np.gradient(self.current_grid[:,:,1], axis=1)
        divergence = div_x + div_y
        
        # Correction du champ de vitesse
        potential = np.zeros_like(divergence)
        for _ in range(10):  # Itérations de relaxation
            laplacian = np.gradient(np.gradient(potential, axis=0), axis=0) + \
                       np.gradient(np.gradient(potential, axis=1), axis=1)
            potential += 0.1 * (divergence - laplacian)
            
        # Application de la correction
        grad_x = np.gradient(potential, axis=0)
        grad_y = np.gradient(potential, axis=1)
        self.current_grid[:,:,0] -= grad_x
        self.current_grid[:,:,1] -= grad_y
        
    def _update_temperature(self, dt: float):
        """Met à jour la distribution de température"""
        # Diffusion thermique
        self.temperature_grid = self._diffuse(
            self.temperature_grid,
            dt * 0.1  # Coefficient de diffusion thermique
        )
        
        # Stratification naturelle
        depth_gradient = np.linspace(0, -5, self.config.grid_size)
        self.temperature_grid = 0.99 * self.temperature_grid + \
                              0.01 * (self.config.temperature + depth_gradient[:, np.newaxis])
                              
    def _update_oxygen(self, dt: float):
        """Met à jour la distribution d'oxygène"""
        # Diffusion
        self.oxygen_grid = self._diffuse(
            self.oxygen_grid,
            dt * self.config.oxygen_diffusion_rate
        )
        
        # Production/consommation
        light_intensity = self._calculate_light_intensity()
        
        # Photosynthèse près de la surface
        surface_production = np.zeros_like(self.oxygen_grid)
        surface_production[:int(self.config.grid_size * 0.3)] = \
            light_intensity * 0.1 * dt
            
        self.oxygen_grid += surface_production
        
        # Consommation naturelle
        self.oxygen_grid *= (1.0 - 0.01 * dt)
        
    def _update_nutrients(self, dt: float):
        """Met à jour la distribution des nutriments"""
        # Diffusion
        self.nutrient_grid = self._diffuse(
            self.nutrient_grid,
            dt * self.config.nutrient_diffusion_rate
        )
        
        # Remontée des nutriments par les courants
        nutrient_transport = np.zeros_like(self.nutrient_grid)
        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                if (i, j) in self.zones[ZoneType.CURRENT]:
                    nutrient_transport[i, j] = 0.1 * dt
                    
        self.nutrient_grid += nutrient_transport
        
        # Maintien d'un gradient naturel
        depth_factor = np.linspace(0.8, 1.2, self.config.grid_size)
        self.nutrient_grid = 0.99 * self.nutrient_grid + \
                            0.01 * depth_factor[:, np.newaxis]
                            
    def _calculate_light_intensity(self) -> float:
        """Calcule l'intensité lumineuse actuelle"""
        # Variation jour/nuit
        day_factor = np.sin(self.day_phase * 2 * np.pi)
        day_factor = max(0, day_factor)  # Pas de lumière négative
        
        # Variation saisonnière
        season_factor = 0.8 + 0.2 * np.sin(self.season_phase * 2 * np.pi)
        
        # Intensité finale
        base_intensity = self.config.light_intensity_range[0]
        intensity_range = (self.config.light_intensity_range[1] - 
                         self.config.light_intensity_range[0])
        
        return base_intensity + intensity_range * day_factor * season_factor
        
    def _diffuse(self, grid: np.ndarray, rate: float) -> np.ndarray:
        """Applique la diffusion à une grille"""
        # Noyau de diffusion
        kernel = np.array([[0.05, 0.2, 0.05],
                          [0.2,  0.0, 0.2],
                          [0.05, 0.2, 0.05]])
        
        # Application de la diffusion
        diffused = np.zeros_like(grid)
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                neighborhood = grid[i-1:i+2, j-1:j+2]
                diffused[i,j] = np.sum(neighborhood * kernel)
        
        return grid * (1 - rate) + diffused * rate
        
    def get_conditions_at(self, position: Tuple[float, float]) -> Dict:
        """Retourne les conditions environnementales à une position donnée"""
        # Conversion en indices de grille
        grid_x = int(position[0] / self.grid_width)
        grid_y = int(position[1] / self.grid_height)
        
        # Limites de la grille
        grid_x = np.clip(grid_x, 0, self.config.grid_size-1)
        grid_y = np.clip(grid_y, 0, self.config.grid_size-1)
        
        return {
            'temperature': self.temperature_grid[grid_y, grid_x],
            'oxygen_level': self.oxygen_grid[grid_y, grid_x],
            'oxygen': self.oxygen_grid[grid_y, grid_x],
            'nutrients': self.nutrient_grid[grid_y, grid_x],
            'water_current': tuple(self.current_grid[grid_y, grid_x]),
            'current': self.current_grid[grid_y, grid_x],
            'light_intensity': self._calculate_light_attenuation(grid_y),
            'depth': grid_y / self.config.grid_size * self.config.height
        }
        
    def _calculate_light_attenuation(self, depth_index: int) -> float:
        """Calcule l'atténuation de la lumière avec la profondeur"""
        surface_intensity = self._calculate_light_intensity()
        depth_factor = np.exp(-0.1 * depth_index)  # Loi de Beer-Lambert simplifiée
        return surface_intensity * depth_factor
        
    def get_flow_field(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Calcule le champ de vitesse à une position donnée"""
        # Utilisation du cache si disponible
        cache_key = (int(position[0]/10), int(position[1]/10))
        if cache_key in self.flow_field_cache:
            return self.flow_field_cache[cache_key]
            
        # Indices de grille
        grid_x = int(position[0] / self.grid_width)
        grid_y = int(position[1] / self.grid_height)
        grid_x = int(np.clip(grid_x, 0, self.config.grid_size - 1))
        grid_y = int(np.clip(grid_y, 0, self.config.grid_size - 1))
        
        # Interpolation bilinéaire du champ de vitesse
        x_frac = (position[0] / self.grid_width) - grid_x
        y_frac = (position[1] / self.grid_height) - grid_y
        
        # Points de la grille
        p00 = self.current_grid[grid_y, grid_x]
        p10 = self.current_grid[grid_y, min(grid_x+1, self.config.grid_size-1)]
        p01 = self.current_grid[min(grid_y+1, self.config.grid_size-1), grid_x]
        p11 = self.current_grid[min(grid_y+1, self.config.grid_size-1),
                               min(grid_x+1, self.config.grid_size-1)]
        
        # Interpolation
        flow = (p00 * (1-x_frac) * (1-y_frac) +
                p10 * x_frac * (1-y_frac) +
                p01 * (1-x_frac) * y_frac +
                p11 * x_frac * y_frac)
        
        # Mise en cache
        self.flow_field_cache[cache_key] = tuple(flow)
        return tuple(flow)
        
    def apply_force_field(self, position: Tuple[float, float], 
                         force: Tuple[float, float], radius: float):
        """Applique une force au champ de fluide"""
        # Conversion en indices de grille
        center_x = int(position[0] / self.grid_width)
        center_y = int(position[1] / self.grid_height)
        grid_radius = max(1, int(radius / self.grid_width))
        
        # Application de la force avec atténuation radiale
        for dy in range(-grid_radius, grid_radius + 1):
            for dx in range(-grid_radius, grid_radius + 1):
                x = center_x + dx
                y = center_y + dy
                
                if 0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size:
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance <= grid_radius:
                        # Atténuation gaussienne
                        influence = np.exp(-0.5 * (distance/grid_radius)**2)
                        self.current_grid[y, x] += np.array(force) * influence
                        
        # Maintien des contraintes physiques
        self._enforce_incompressibility()

    def _handle_tentacle_interaction(self, org1: 'CnidarianOrganism', 
                                   org2: 'CnidarianOrganism', 
                                   distance: float, intensity: float):
        """Gère les interactions entre tentacules"""
        # Vérification des zones d'interaction des tentacules
        for tentacle1 in org1.tentacles:
            for tentacle2 in org2.tentacles:
                for segment1 in tentacle1.segments:
                    for segment2 in tentacle2.segments:
                        # Distance entre segments
                        seg_dist = np.sqrt(
                            (segment1.base_x - segment2.base_x)**2 + 
                            (segment1.base_y - segment2.base_y)**2
                        )
                        
                        if seg_dist < self.config.min_interaction_distance:
                            # Activation des cnidocytes
                            for cnido1 in segment1.cnidocytes:
                                if cnido1.is_charged:
                                    cnido1.update(0.1, 1.0)  # Déclenchement
                                    segment1.local_nerve_net.stimulate_region(
                                        cnido1.x, cnido1.y,
                                        radius=5.0,
                                        strength=0.8
                                    )
                                    
                            # Répulsion mécanique des segments
                            repulsion = self.config.repulsion_strength * intensity
                            angle = np.arctan2(
                                segment2.base_y - segment1.base_y,
                                segment2.base_x - segment1.base_x
                            )
                            segment1.velocity -= repulsion * np.array([np.cos(angle), np.sin(angle)])
                            segment2.velocity += repulsion * np.array([np.cos(angle), np.sin(angle)])

    def _handle_neural_interaction(self, org1: 'CnidarianOrganism', 
                                 org2: 'CnidarianOrganism',
                                 intensity: float):
        """Gère les interactions entre réseaux nerveux"""
        # Transmission des signaux nerveux entre organismes proches
        if org1.nerve_net and org2.nerve_net:
            # Calcul des zones de chevauchement des réseaux
            for neuron1 in org1.nerve_net.neurons:
                for neuron2 in org2.nerve_net.neurons:
                    dist = np.sqrt(
                        (neuron1.x - neuron2.x)**2 + 
                        (neuron1.y - neuron2.y)**2
                    )
                    
                    if dist < self.config.min_interaction_distance:
                        # Couplage nerveux
                        if neuron1.is_active:
                            neuron2.membrane_potential += 5.0 * intensity
                        if neuron2.is_active:
                            neuron1.membrane_potential += 5.0 * intensity

    def _handle_digestive_interaction(self, org1: 'CnidarianOrganism', 
                                    org2: 'CnidarianOrganism',
                                    distance: float):
        """Gère les interactions entre systèmes digestifs"""
        # Transfert de nutriments si les cavités gastriques sont proches
        if (hasattr(org1, 'gastric_cavity') and 
            hasattr(org2, 'gastric_cavity')):
            
            if distance < org1.radius + org2.radius + 10:
                # Échange de nutriments proportionnel au gradient
                nutrient_diff = (org1.gastric_cavity.nutrient_level - 
                               org2.gastric_cavity.nutrient_level)
                
                transfer = nutrient_diff * 0.1
                org1.gastric_cavity.nutrient_level -= transfer
                org2.gastric_cavity.nutrient_level += transfer

    def _cleanup_caches(self):
        """Nettoie les caches d'interaction"""
        current_time = time.time()
        
        # Nettoyage des collisions récentes
        self.recent_collisions.clear()
        
        # Nettoyage des cooldowns expirés
        expired = []
        for pair_id, timestamp in self.interaction_cooldowns.items():
            if current_time - timestamp > 2.0:  # 2 secondes de cooldown
                expired.append(pair_id)
                
        for pair_id in expired:
            del self.interaction_cooldowns[pair_id]

    def get_stats(self) -> Dict:
        """Retourne les statistiques du système d'interactions"""
        return {
            'collision_count': self.collision_count,
            'interaction_count': self.interaction_count,
            'active_cooldowns': len(self.interaction_cooldowns),
            'recent_collisions': len(self.recent_collisions),
            'update_rate': 1.0 / max(0.001, time.time() - self.last_update_time)
        }

    def apply_boundary_constraints(self, organism: 'CnidarianOrganism'):
        """Applique les contraintes de bord de l'environnement"""
        # Rebond sur les bords
        margin = organism.radius * 1.1
        
        if organism.x < margin:
            organism.x = margin
            organism.velocity[0] = abs(organism.velocity[0]) * self.config.collision_elasticity
            
        elif organism.x > self.width - margin:
            organism.x = self.width - margin
            organism.velocity[0] = -abs(organism.velocity[0]) * self.config.collision_elasticity
            
        if organism.y < margin:
            organism.y = margin
            organism.velocity[1] = abs(organism.velocity[1]) * self.config.collision_elasticity
            
        elif organism.y > self.height - margin:
            organism.y = self.height - margin
            organism.velocity[1] = -abs(organism.velocity[1]) * self.config.collision_elasticity
