import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import time
from src.config.settings import sim_config

@dataclass
class InteractionConfig:
    """Configuration du système d'interactions"""
    # Paramètres de collision
    collision_elasticity: float = 0.5
    friction_coefficient: float = 0.3
    repulsion_strength: float = 2.0
    
    # Paramètres de quadrillage spatial
    grid_cell_size: float = 50.0  # Taille des cellules pour la détection
    grid_optimization: bool = True # Utiliser le quadrillage spatial
    
    # Paramètres d'interaction
    interaction_radius: float = 100.0
    max_interactions_per_frame: int = 100
    
    # Seuils
    min_impact_force: float = 0.1
    min_interaction_distance: float = 5.0

class SpatialGrid:
    """Grille spatiale pour optimisation des détections de collision"""
    def __init__(self, width: float, height: float, cell_size: float):
        self.cell_size = cell_size
        self.cols = int(width / cell_size) + 1
        self.rows = int(height / cell_size) + 1
        self.grid = {}  # Dict[(int, int), Set[CnidarianOrganism]]
        
    def clear(self):
        """Vide la grille"""
        self.grid.clear()
        
    def insert(self, organism: 'CnidarianOrganism'):
        """Insère un organisme dans les cellules appropriées"""
        # Calcul des cellules couvertes par l'organisme
        min_x = int((organism.x - organism.radius) / self.cell_size)
        max_x = int((organism.x + organism.radius) / self.cell_size)
        min_y = int((organism.y - organism.radius) / self.cell_size)
        max_y = int((organism.y + organism.radius) / self.cell_size)
        
        # Insertion dans chaque cellule
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                if (i, j) not in self.grid:
                    self.grid[(i, j)] = set()
                self.grid[(i, j)].add(organism)
                
    def get_nearby(self, organism: 'CnidarianOrganism') -> Set['CnidarianOrganism']:
        """Retourne les organismes potentiellement en collision"""
        nearby = set()
        
        # Calcul des cellules à vérifier
        min_x = int((organism.x - organism.radius) / self.cell_size)
        max_x = int((organism.x + organism.radius) / self.cell_size)
        min_y = int((organism.y - organism.radius) / self.cell_size)
        max_y = int((organism.y + organism.radius) / self.cell_size)
        
        # Collecte des organismes dans les cellules
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                if (i, j) in self.grid:
                    nearby.update(self.grid[(i, j)])
                    
        nearby.discard(organism)  # Retire l'organisme lui-même
        return nearby

class InteractionSystem:
    """Gestion des collisions et interactions entre cnidaires"""
    def __init__(self, width: float, height: float, 
                 config: Optional[InteractionConfig] = None):
        self.config = config or InteractionConfig()
        self.width = width
        self.height = height
        
        # Grille spatiale pour optimisation
        self.spatial_grid = SpatialGrid(
            width, height,
            self.config.grid_cell_size
        ) if self.config.grid_optimization else None
        
        # Cache des interactions récentes
        self.recent_collisions: Set[Tuple[int, int]] = set()
        self.interaction_cooldowns: Dict[Tuple[int, int], float] = {}
        
        # Statistiques
        self.collision_count = 0
        self.interaction_count = 0
        self.last_update_time = time.time()
        
    def update(self, organisms: List['CnidarianOrganism'], dt: float):
        """Met à jour toutes les interactions"""
        # Mise à jour de la grille spatiale
        if self.spatial_grid:
            self.spatial_grid.clear()
            for organism in organisms:
                self.spatial_grid.insert(organism)
                
        # Traitement des collisions et interactions
        self._process_interactions(organisms, dt)
        
        # Nettoyage des caches
        self._cleanup_caches()
        
    def _process_interactions(self, organisms: List['CnidarianOrganism'], dt: float):
        """Traite toutes les interactions entre organismes"""
        processed_pairs = set()
        
        for i, org1 in enumerate(organisms):
            # Obtention des organismes proches
            nearby = (self.spatial_grid.get_nearby(org1) if self.spatial_grid 
                     else organisms[i+1:])
            
            for org2 in nearby:
                pair_id = tuple(sorted([id(org1), id(org2)]))
                if pair_id in processed_pairs:
                    continue
                    
                # Vérification de la distance
                distance = np.sqrt(
                    (org1.x - org2.x)**2 + 
                    (org1.y - org2.y)**2
                )
                
                # Collision physique
                if distance < (org1.radius + org2.radius):
                    self._handle_collision(org1, org2, distance, dt)
                    
                # Interactions biologiques
                elif distance < self.config.interaction_radius:
                    self._handle_interaction(org1, org2, distance, dt)
                    
                processed_pairs.add(pair_id)
                
    def _handle_collision(self, org1: 'CnidarianOrganism', 
                         org2: 'CnidarianOrganism',
                         distance: float, dt: float):
        """Gère la collision physique entre deux organismes"""
        if distance <= 1e-6:
            distance = 1e-6

        # Vecteur de collision normalisé
        nx = (org2.x - org1.x) / distance
        ny = (org2.y - org1.y) / distance
        
        # Chevauchement
        overlap = (org1.radius + org2.radius) - distance
        
        # Forces de répulsion
        repulsion_force = overlap * self.config.repulsion_strength
        
        # Application des forces en fonction de la masse
        total_mass = org1.mass + org2.mass
        org1_ratio = org2.mass / total_mass
        org2_ratio = org1.mass / total_mass
        
        # Déplacement pour éviter le chevauchement
        org1.x -= nx * overlap * org1_ratio
        org1.y -= ny * overlap * org1_ratio
        org2.x += nx * overlap * org2_ratio
        org2.y += ny * overlap * org2_ratio
        
        # Calcul des vitesses relatives
        vx = org2.velocity[0] - org1.velocity[0]
        vy = org2.velocity[1] - org1.velocity[1]
        
        # Vitesse normale
        vn = vx * nx + vy * ny
        
        # S'ils s'éloignent déjà, pas besoin de calculer l'impulsion
        if vn > 0:
            return
            
        # Calcul de l'impulsion
        impulse = -(1 + self.config.collision_elasticity) * vn
        impulse_x = impulse * nx
        impulse_y = impulse * ny
        
        # Application des impulsions
        org1.velocity[0] -= impulse_x * org1_ratio
        org1.velocity[1] -= impulse_y * org1_ratio
        org2.velocity[0] += impulse_x * org2_ratio
        org2.velocity[1] += impulse_y * org2_ratio
        
        # Application de la friction
        friction = self.config.friction_coefficient * np.abs(impulse)
        
        # Composantes tangentielles
        tx = -ny  # Vecteur tangent
        ty = nx
        vt = vx * tx + vy * ty  # Vitesse tangentielle
        
        # Application de la friction
        if np.abs(vt) > 0:
            friction_impulse = np.clip(friction, 0, np.abs(vt))
            friction_dir = -1 if vt > 0 else 1
            
            # Application de la friction aux vitesses
            org1.velocity[0] += friction_dir * friction_impulse * tx * org1_ratio
            org1.velocity[1] += friction_dir * friction_impulse * ty * org1_ratio
            org2.velocity[0] -= friction_dir * friction_impulse * tx * org2_ratio
            org2.velocity[1] -= friction_dir * friction_impulse * ty * org2_ratio
            
        # Notification aux organismes
        impact_force = np.sqrt(impulse_x**2 + impulse_y**2)
        if impact_force > self.config.min_impact_force:
            org1.on_collision(org2, impact_force)
            org2.on_collision(org1, impact_force)
            
            # Enregistrement de la collision
            self.collision_count += 1
            collision_id = tuple(sorted([id(org1), id(org2)]))
            self.recent_collisions.add(collision_id)
            
    def _handle_interaction(self, org1: 'CnidarianOrganism', 
                          org2: 'CnidarianOrganism',
                          distance: float, dt: float):
        """Gère les interactions biologiques entre organismes"""
        # Vérification du cooldown
        interaction_id = tuple(sorted([id(org1), id(org2)]))
        current_time = time.time()
        
        if interaction_id in self.interaction_cooldowns:
            if current_time - self.interaction_cooldowns[interaction_id] < 1.0:
                return
                
        # Calcul de l'intensité de l'interaction
        intensity = 1.0 - (distance / self.config.interaction_radius)
        
        # Interaction des tentacules
        self._handle_tentacle_interaction(org1, org2, distance, intensity)
        
        # Interaction des systèmes nerveux
        self._handle_neural_interaction(org1, org2, intensity)
        
        # Interaction des systèmes digestifs
        self._handle_digestive_interaction(org1, org2, distance)
        
        # Mise à jour du cooldown
        self.interaction_cooldowns[interaction_id] = current_time
        self.interaction_count += 1

    def _handle_tentacle_interaction(self, org1: 'CnidarianOrganism',
                                   org2: 'CnidarianOrganism',
                                   distance: float, intensity: float):
        """Stimule localement les tentacules de deux organismes proches."""
        interaction_radius = self.config.min_interaction_distance * 4

        for tentacle1 in org1.tentacles:
            for tentacle2 in org2.tentacles:
                base_distance = np.sqrt(
                    (tentacle1.base_x - tentacle2.base_x) ** 2 +
                    (tentacle1.base_y - tentacle2.base_y) ** 2
                )
                if base_distance >= interaction_radius:
                    continue

                threat_strength = min(1.0, intensity + 0.2)
                tentacle1.local_nerve_net.stimulate_region(
                    tentacle1.base_x,
                    tentacle1.base_y,
                    radius=10.0,
                    strength=threat_strength
                )
                tentacle2.local_nerve_net.stimulate_region(
                    tentacle2.base_x,
                    tentacle2.base_y,
                    radius=10.0,
                    strength=threat_strength
                )

                for segment in tentacle1.segments:
                    for cnidocyte in segment.cnidocytes:
                        cnidocyte.update(0.1, threat_strength)

                for segment in tentacle2.segments:
                    for cnidocyte in segment.cnidocytes:
                        cnidocyte.update(0.1, threat_strength)

    def _handle_neural_interaction(self, org1: 'CnidarianOrganism',
                                 org2: 'CnidarianOrganism',
                                 intensity: float):
        """Couple faiblement l'activité nerveuse de deux organismes proches."""
        if not org1.nerve_net or not org2.nerve_net:
            return

        coupling_strength = 5.0 * intensity
        if org1.nerve_net.global_activity > 0.2:
            org2.nerve_net.stimulate_region(org2.x, org2.y, radius=15.0, strength=coupling_strength)
        if org2.nerve_net.global_activity > 0.2:
            org1.nerve_net.stimulate_region(org1.x, org1.y, radius=15.0, strength=coupling_strength)

    def _handle_digestive_interaction(self, org1: 'CnidarianOrganism',
                                    org2: 'CnidarianOrganism',
                                    distance: float):
        """Équilibre légèrement les nutriments entre organismes très proches."""
        if not hasattr(org1, 'gastric_cavity') or not hasattr(org2, 'gastric_cavity'):
            return

        if distance >= org1.radius + org2.radius + 10:
            return

        nutrient_diff = org1.gastric_cavity.nutrient_level - org2.gastric_cavity.nutrient_level
        transfer = nutrient_diff * 0.1
        org1.gastric_cavity.nutrient_level -= transfer
        org2.gastric_cavity.nutrient_level += transfer

    def _cleanup_caches(self):
        """Nettoie les caches d'interaction."""
        current_time = time.time()
        self.recent_collisions.clear()

        expired = []
        for pair_id, timestamp in self.interaction_cooldowns.items():
            if current_time - timestamp > 2.0:
                expired.append(pair_id)

        for pair_id in expired:
            del self.interaction_cooldowns[pair_id]

    def get_stats(self) -> Dict:
        """Retourne des métriques simples du système d'interactions."""
        return {
            'collision_count': self.collision_count,
            'interaction_count': self.interaction_count,
            'active_cooldowns': len(self.interaction_cooldowns),
            'recent_collisions': len(self.recent_collisions),
            'update_rate': 1.0 / max(0.001, time.time() - self.last_update_time)
        }

    def apply_boundary_constraints(self, organism: 'CnidarianOrganism'):
        """Empêche un organisme de sortir de la zone simulée."""
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
