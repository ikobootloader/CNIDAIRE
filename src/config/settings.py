"""
Configuration centrale pour le projet Cnidaire 2D.
Gère les paramètres globaux et les constantes de simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import os

@dataclass
class SimulationConfig:
    """Configuration globale de la simulation"""
    # Paramètres temporels
    TIMESTEP: float = 0.016  # ~60 FPS
    MAX_FRAME_SKIP: int = 5
    
    # Dimensions de l'environnement
    WORLD_WIDTH: float = 1200.0
    WORLD_HEIGHT: float = 800.0
    
    # Paramètres physiques
    GRAVITY: float = 0.0  # Pas de gravité dans l'eau
    WATER_DENSITY: float = 1.0
    WATER_VISCOSITY: float = 0.001
    
    # Paramètres biologiques de base
    MIN_CNIDARIAN_SIZE: float = 10.0
    MAX_CNIDARIAN_SIZE: float = 100.0
    BASE_METABOLISM_RATE: float = 0.01
    
    # Paramètres environnementaux
    TEMPERATURE_RANGE: Tuple[float, float] = (15.0, 25.0)
    OPTIMAL_TEMPERATURE: float = 20.0
    DAY_LENGTH: float = 300.0  # en secondes
    SEASON_LENGTH: float = 1200.0  # en secondes

@dataclass
class RenderConfig:
    """Configuration du rendu graphique"""
    # Paramètres de fenêtre
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 800
    TARGET_FPS: int = 60
    
    # Couleurs (R,G,B)
    COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'background': (10, 20, 40),
        'epidermis': (200, 220, 255),
        'mesoglea': (180, 200, 240),
        'nerve': (255, 100, 100),
        'debug': (255, 255, 0)
    })
    
    # Options de rendu
    ENABLE_ANTIALIASING: bool = True
    ENABLE_VSYNC: bool = True
    SHOW_DEBUG: bool = False
    
class Paths:
    """Gestion des chemins du projet"""
    # Chemins de base
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC = os.path.join(ROOT, 'src')
    ASSETS = os.path.join(ROOT, 'assets')
    DATA = os.path.join(ROOT, 'data')
    
    # Sous-dossiers
    MODELS = os.path.join(SRC, 'models')
    PHYSICS = os.path.join(SRC, 'physics')
    ENVIRONMENT = os.path.join(SRC, 'environment')
    VISUALIZATION = os.path.join(SRC, 'visualization')
    
    @classmethod
    def ensure_directories(cls):
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        for path in [cls.ASSETS, cls.DATA]:
            if not os.path.exists(path):
                os.makedirs(path)

# Configuration globale par défaut
sim_config = SimulationConfig()
render_config = RenderConfig()

# Création des répertoires nécessaires
Paths.ensure_directories()