from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from src.environment.aquatic_environment import Environment, EnvironmentConfig
from src.models.cnidarian_organism import CnidarianOrganism
from src.physics.interaction_system import InteractionSystem, InteractionConfig


@dataclass
class SimulationConfig:
    """Configuration légère de l'orchestrateur principal."""
    organism_count: int = 3
    prey_count: int = 12
    prey_spawn_interval: float = 2.0
    dt: float = 0.1
    random_seed: Optional[int] = 42


class CnidarianSimulation:
    """Orchestre environnement, organismes, proies et interactions."""
    def __init__(self,
                 simulation_config: Optional[SimulationConfig] = None,
                 environment_config: Optional[EnvironmentConfig] = None,
                 interaction_config: Optional[InteractionConfig] = None):
        self.config = simulation_config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.environment = Environment(environment_config)
        self.interactions = InteractionSystem(
            self.environment.config.width,
            self.environment.config.height,
            interaction_config
        )
        self.time = 0.0
        self.time_since_prey_spawn = 0.0
        self.next_prey_id = 0
        self.organisms = self._initialize_organisms()
        self.prey = self._initialize_prey()

    def _initialize_organisms(self) -> List[CnidarianOrganism]:
        organisms = []
        spacing = self.environment.config.width / (self.config.organism_count + 1)

        for index in range(self.config.organism_count):
            x = spacing * (index + 1)
            y = self.environment.config.height * (0.35 + 0.3 * (index % 2))
            organisms.append(CnidarianOrganism(x=x, y=y))

        return organisms

    def _initialize_prey(self) -> List[Dict]:
        return [self._create_prey() for _ in range(self.config.prey_count)]

    def _create_prey(self) -> Dict:
        width = self.environment.config.width
        height = self.environment.config.height
        prey = {
            'id': self.next_prey_id,
            'position': (
                float(self.rng.uniform(40, width - 40)),
                float(self.rng.uniform(40, height - 40))
            ),
            'velocity': np.array([
                self.rng.uniform(-8.0, 8.0),
                self.rng.uniform(-8.0, 8.0)
            ], dtype=float),
            'parameters': {
                'size': float(self.rng.uniform(3.0, 12.0)),
                'energy_content': float(self.rng.uniform(8.0, 20.0)),
                'resistance': float(self.rng.uniform(0.1, 0.5)),
                'escape_strength': float(self.rng.uniform(0.2, 0.8)),
                'digestion_difficulty': float(self.rng.uniform(1.0, 4.0))
            }
        }
        self.next_prey_id += 1
        return prey

    def step(self, dt: Optional[float] = None) -> Dict:
        """Exécute un pas complet de simulation."""
        dt = dt or self.config.dt
        self.time += dt
        self.time_since_prey_spawn += dt

        self.environment.update(dt)
        self._update_prey(dt)

        offspring = []
        for organism in self.organisms:
            local_conditions = self.environment.get_conditions_at((organism.x, organism.y))
            local_conditions['prey'] = self._get_visible_prey(organism)
            offspring.extend(organism.update(dt, local_conditions))

        self.interactions.update(self.organisms, dt)
        for organism in self.organisms:
            self.interactions.apply_boundary_constraints(organism)

        if offspring:
            self.organisms.extend(offspring)

        self._remove_consumed_prey()
        self._replenish_prey()
        return self.get_state_snapshot()

    def run_steps(self, steps: int, dt: Optional[float] = None) -> Dict:
        """Exécute plusieurs pas de simulation."""
        snapshot = self.get_state_snapshot()
        for _ in range(steps):
            snapshot = self.step(dt)
        return snapshot

    def get_state_snapshot(self) -> Dict:
        """Retourne un état synthétique de la simulation."""
        return {
            'time': self.time,
            'organism_count': len(self.organisms),
            'prey_count': len(self.prey),
            'prey': [
                {
                    'world_id': prey['id'],
                    'position': prey['position'],
                    'parameters': prey['parameters']
                }
                for prey in self.prey
            ],
            'flow_samples': self._build_flow_samples(),
            'average_energy': float(
                sum(organism.energy for organism in self.organisms) / max(1, len(self.organisms))
            ),
            'average_health': float(
                sum(organism.health for organism in self.organisms) / max(1, len(self.organisms))
            ),
            'interaction_stats': self.interactions.get_stats()
        }

    def _update_prey(self, dt: float):
        width = self.environment.config.width
        height = self.environment.config.height

        for prey in self.prey:
            flow = np.array(self.environment.get_flow_field(prey['position']), dtype=float)
            prey['velocity'] += flow * 0.05 * dt
            prey['position'] = (
                float(prey['position'][0] + prey['velocity'][0] * dt),
                float(prey['position'][1] + prey['velocity'][1] * dt)
            )

            x, y = prey['position']
            vx, vy = prey['velocity']

            if x < 0 or x > width:
                vx *= -1
                x = float(np.clip(x, 0, width))
            if y < 0 or y > height:
                vy *= -1
                y = float(np.clip(y, 0, height))

            prey['position'] = (x, y)
            prey['velocity'] = np.array([vx, vy], dtype=float)

    def _get_visible_prey(self, organism: CnidarianOrganism) -> List[Dict]:
        visible = []
        detection_radius = max(organism.radius * 3, 120.0)

        for prey in self.prey:
            distance = np.sqrt(
                (prey['position'][0] - organism.x) ** 2 +
                (prey['position'][1] - organism.y) ** 2
            )
            if distance <= detection_radius:
                visible.append({
                    'world_prey_id': prey['id'],
                    'position': prey['position'],
                    'parameters': prey['parameters']
                })

        return visible

    def _remove_consumed_prey(self):
        ingested_world_ids = set()

        for organism in self.organisms:
            ingested_world_ids.update(organism.feeding_system.pop_ingested_world_prey_ids())

        if not ingested_world_ids:
            return

        self.prey = [prey for prey in self.prey if prey['id'] not in ingested_world_ids]

    def _replenish_prey(self):
        if self.time_since_prey_spawn >= self.config.prey_spawn_interval:
            self.time_since_prey_spawn = 0.0
            self.prey.append(self._create_prey())

        while len(self.prey) < self.config.prey_count:
            self.prey.append(self._create_prey())

    def _build_flow_samples(self) -> List[Dict]:
        """Construit un petit échantillon visuel des courants."""
        samples = []
        width = self.environment.config.width
        height = self.environment.config.height

        for x in range(120, int(width), 220):
            for y in range(120, int(height), 180):
                samples.append({
                    'position': (float(x), float(y)),
                    'flow': self.environment.get_flow_field((float(x), float(y)))
                })

        return samples
