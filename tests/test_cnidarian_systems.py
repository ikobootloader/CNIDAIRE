import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnidarian_organism import CnidarianOrganism
from src.visualization.renderer import CnidarianRenderer, VisualizationConfig
from src.models.growth_system import GrowthStage
import numpy as np
import time
import pygame

def assert_in_range(value: float, min_val: float, max_val: float, name: str):
    """Vérifie qu'une valeur est dans une plage donnée"""
    assert min_val <= value <= max_val, f"{name} hors limites: {value} (attendu entre {min_val} et {max_val})"

def test_growth_and_immune_response():
    """Test basique des systèmes de croissance et immunitaire"""
    # Initialisation du renderer
    vis_config = VisualizationConfig()
    renderer = CnidarianRenderer(vis_config)
    
    # Création d'un cnidaire
    cnidarian = CnidarianOrganism(x=vis_config.window_width//2, y=vis_config.window_height//2)
    
    # Réduire la durée juvenile pour le test
    cnidarian.growth_system.params.juvenile_duration = 300.0  # 5 minutes
    
    # Simulation sur plusieurs pas de temps
    dt = 0.1
    total_time = 0
    threats_handled = 0
    immune_activations = []
    running = True
    clock = pygame.time.Clock()
    
    print("\nDébut du test de simulation\n")
    
    initial_size = cnidarian.growth_system.size
    initial_energy = cnidarian.energy
    
    # Génération d'une menace immédiate pour tester le système immunitaire
    threat_position = (cnidarian.x + 10, cnidarian.y + 10)
    threat_response = cnidarian.handle_threat('pathogen', threat_position, 0.7)
    
    if threat_response:
        threats_handled += 1
        immune_activations.append({
            'time': 0,
            'pre_activation': 0,
            'post_activation': cnidarian.immune_system.global_activation,
            'response': True,
            'active_responses': len(cnidarian.immune_system.active_responses)
        })
        
    while running and total_time < 60.0:  # 60 secondes de simulation
        # Gestion des événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        total_time += dt
        
        # Conditions environnementales simulées
        environment_state = {
            'temperature': 20.0 + np.sin(total_time) * 2,
            'nutrients': 0.8 + np.random.normal(0, 0.1),
            'oxygen_level': 0.9
        }
        
        # Mise à jour de l'organisme
        cnidarian.update(dt, environment_state)
        
        # Test du système immunitaire toutes les secondes
        if int(total_time) > int(total_time - dt):
            threat_position = (
                cnidarian.x + np.random.normal(0, 10),
                cnidarian.y + np.random.normal(0, 10)
            )
            
            pre_activation = cnidarian.immune_system.global_activation
            response = cnidarian.handle_threat('pathogen', threat_position, 0.7)
            post_activation = cnidarian.immune_system.global_activation
            
            if response:
                threats_handled += 1
            
            immune_activations.append({
                'time': total_time,
                'pre_activation': pre_activation,
                'post_activation': post_activation,
                'response': response,
                'active_responses': len(cnidarian.immune_system.active_responses)
            })
            
            print(f"\nTemps: {total_time:.1f}s")
            print(f"Taille: {cnidarian.growth_system.size:.2f}")
            print(f"Stade: {cnidarian.developmental_stage}")
            print(f"Vitalité: {cnidarian.vitality:.2f}")
            print(f"Énergie: {cnidarian.energy:.2f}")
            print(f"Santé: {cnidarian.health:.2f}")
            print(f"Activation immunitaire: {post_activation:.3f}")
            print(f"Réponses immunitaires actives: {len(cnidarian.immune_system.active_responses)}")
            print("-" * 40)
            
        # Rendu de la simulation
        renderer.render([cnidarian], environment_state)
        clock.tick(60)  # Limite à 60 FPS
        
    # Nettoyage Pygame
    renderer.cleanup()
    
    # Affichage du résumé des activations immunitaires
    print("\nRésumé des activations immunitaires:")
    print(f"Nombre total de menaces: {threats_handled}")
    print(f"Nombre d'activations enregistrées: {len(immune_activations)}")
    
    print("\nDétail des dernières activations:")
    for activation in immune_activations[-3:]:
        print(f"t={activation['time']:.1f}s: "
              f"pré={activation['pre_activation']:.3f} -> "
              f"post={activation['post_activation']:.3f} "
              f"(réponse: {activation['response']}, "
              f"actives: {activation['active_responses']})")
    
    print("\nÉtat final du développement:")
    print(f"Age actuel: {cnidarian.age:.2f}s")
    print(f"Stade actuel: {cnidarian.developmental_stage}")
    print(f"Durée juvénile configurée: {cnidarian.growth_system.params.juvenile_duration}s")
    
    # Vérifications finales
    final_checks = {
        "Croissance": cnidarian.growth_system.size > initial_size,
        "Énergie consommée": cnidarian.energy < initial_energy,
        "Système immunitaire actif": any(a['post_activation'] > 0 for a in immune_activations),
        "Stade développemental correct": cnidarian.developmental_stage == "juvenile"
    }
    
    for check_name, result in final_checks.items():
        if not result:
            print(f"\nÉchec de la vérification : {check_name}")
            if check_name == "Stade développemental correct":
                print(f"Stade attendu: juvenile")
                print(f"Stade actuel: {cnidarian.developmental_stage}")
        assert result, f"Échec : {check_name}"
        print(f"✓ {check_name}")

if __name__ == "__main__":
    test_growth_and_immune_response()