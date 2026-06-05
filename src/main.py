import pygame

from src.simulation.cnidarian_simulation import CnidarianSimulation
from src.visualization.renderer import CnidarianRenderer, VisualizationConfig


def main():
    simulation = CnidarianSimulation()
    renderer = CnidarianRenderer(VisualizationConfig())
    running = True
    paused = False

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            if not paused:
                snapshot = simulation.step()
            else:
                snapshot = simulation.get_state_snapshot()

            environment_data = {
                'particle_density': min(1.0, 0.35 + snapshot['prey_count'] / 24.0),
                'prey': snapshot['prey'],
                'flow_samples': snapshot['flow_samples']
            }
            renderer.render(simulation.organisms, environment_data)
    finally:
        renderer.cleanup()


if __name__ == "__main__":
    main()
