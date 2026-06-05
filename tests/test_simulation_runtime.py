import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.aquatic_environment import Environment
from src.models.cnidarian_organism import CnidarianOrganism
from src.models.feeding_system import PreyParameters, PreyState
from src.models.reproduction_system import ReproductionParameters, ReproductionMode
from src.physics.interaction_system import InteractionSystem
from src.simulation.cnidarian_simulation import CnidarianSimulation, SimulationConfig


def test_environment_update_exposes_compatible_conditions():
    environment = Environment()
    environment.update(0.1)

    conditions = environment.get_conditions_at((100.0, 120.0))

    assert 'oxygen_level' in conditions
    assert 'water_current' in conditions
    assert np.isfinite(conditions['temperature'])
    assert len(conditions['water_current']) == 2


def test_feeding_system_completes_ingestion_cycle():
    organism = CnidarianOrganism(200.0, 200.0)
    prey_params = PreyParameters(
        size=5.0,
        energy_content=12.0,
        resistance=0.1,
        escape_strength=0.2,
        digestion_difficulty=1.0
    )

    organism.feeding_system.captured_prey[0] = (
        PreyState.PARALYZED,
        prey_params,
        organism.feeding_system.mouth_position
    )

    organism.feeding_system.update(0.1)
    organism.feeding_system.update(0.1)

    assert 0 not in organism.feeding_system.captured_prey
    assert organism.gastric_cavity.food_content > 0.0


def test_interaction_system_handles_collision_callbacks():
    organism_a = CnidarianOrganism(200.0, 200.0)
    organism_b = CnidarianOrganism(240.0, 200.0)
    organism_a.velocity = np.array([1.0, 0.0], dtype=float)
    organism_b.velocity = np.array([-1.0, 0.0], dtype=float)

    interactions = InteractionSystem(1200.0, 800.0)
    interactions.update([organism_a, organism_b], 0.1)

    assert interactions.collision_count >= 1
    assert organism_a.collision_count >= 1
    assert organism_b.collision_count >= 1


def test_reproduction_system_uses_simulation_time():
    organism = CnidarianOrganism(200.0, 200.0)
    organism.age = 500.0
    organism.energy = 100.0
    organism.health = 1.0
    organism.set_size(120.0)

    params = ReproductionParameters(
        min_age_for_reproduction=0.0,
        reproduction_cooldown=0.0,
        budding_duration=0.2,
        fission_duration=0.2
    )
    organism.reproduction_system.params = params
    organism.reproduction_system.time_since_last_reproduction = params.reproduction_cooldown

    offspring = None
    for _ in range(10):
        offspring = organism.reproduction_system.update(0.1)
        if offspring is not None:
            break

    assert offspring is not None
    assert organism.reproduction_system.current_mode == ReproductionMode.NONE


def test_simulation_orchestrator_runs_multiple_steps():
    simulation = CnidarianSimulation(
        simulation_config=SimulationConfig(
            organism_count=2,
            prey_count=4,
            dt=0.1,
            random_seed=7
        )
    )

    snapshot = simulation.run_steps(5)

    assert snapshot['organism_count'] >= 2
    assert snapshot['prey_count'] >= 1
    assert snapshot['interaction_stats']['interaction_count'] >= 0


def test_simulation_removes_only_ingested_prey():
    simulation = CnidarianSimulation(
        simulation_config=SimulationConfig(
            organism_count=1,
            prey_count=1,
            dt=0.1,
            random_seed=3
        )
    )

    prey = simulation.prey[0]
    organism = simulation.organisms[0]
    organism.feeding_system.captured_prey[0] = (
        PreyState.CAPTURED,
        PreyParameters(**prey['parameters']),
        prey['position'],
        prey['id']
    )

    simulation._remove_consumed_prey()
    assert len(simulation.prey) == 1

    organism.feeding_system.ingested_world_prey_ids.append(prey['id'])
    simulation._remove_consumed_prey()
    assert len(simulation.prey) == 0


def test_swimming_updates_pulsation_state():
    organism = CnidarianOrganism(200.0, 200.0)

    organism._initiate_swimming(0.9, 0.1)

    assert organism.is_swimming is True
    assert organism.body_pulse > 0.0
    assert organism.pulse_frequency > 0.0
    assert np.linalg.norm(organism.velocity) > 0.0


def test_visible_prey_orients_swimming_direction():
    organism = CnidarianOrganism(200.0, 200.0)
    organism.orientation = np.pi / 2

    original_nerve_update = organism.nerve_net.update

    def directed_nerve_update(dt):
        response = original_nerve_update(dt)
        response[:] = 0.2
        response[:len(response)//2] = 0.85
        return response

    organism.nerve_net.update = directed_nerve_update
    prey_payload = [{
        'position': (320.0, 200.0),
        'parameters': {
            'size': 6.0,
            'energy_content': 10.0,
            'resistance': 0.1,
            'escape_strength': 0.2,
            'digestion_difficulty': 1.0
        }
    }]

    organism.update(0.1, {'prey': prey_payload, 'water_current': (0.0, 0.0)})

    assert organism.orientation < np.pi / 2
    assert organism.is_swimming is True


def test_organism_updates_nerve_and_immune_system_once_per_step():
    organism = CnidarianOrganism(200.0, 200.0)
    nerve_calls = {'count': 0}
    immune_calls = {'count': 0}

    original_nerve_update = organism.nerve_net.update
    original_immune_update = organism.immune_system.update

    def counted_nerve_update(dt):
        nerve_calls['count'] += 1
        return original_nerve_update(dt)

    def counted_immune_update(dt, nerve_activity=None):
        immune_calls['count'] += 1
        return original_immune_update(dt, nerve_activity=nerve_activity)

    organism.nerve_net.update = counted_nerve_update
    organism.immune_system.update = counted_immune_update

    organism.update(0.1, {})

    assert nerve_calls['count'] == 1
    assert immune_calls['count'] == 1
