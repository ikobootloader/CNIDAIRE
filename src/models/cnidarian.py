import numpy as np
from typing import List, Tuple, Optional
import pygame

class CnidarianBody:
    """
    Représente la structure anatomique de base d'un cnidaire (méduse) en 2D
    """
    def __init__(self, x: float, y: float, radius: float = 30.0):
        # Position dans l'environnement
        self.x = x
        self.y = y
        self.radius = radius

        # Structure anatomique
        self.bell_elasticity = 0.8  # Élasticité de la cloche (mesoglee)
        self.contraction_state = 0.0  # État de contraction (0 = relaxé, 1 = contracté)
        
        # Couches anatomiques fondamentales
        self.epidermis_thickness = radius * 0.05  # Ectoderme (couche externe)
        self.gastrodermis_thickness = radius * 0.08  # Endoderme (couche interne)
        self.mesoglea_thickness = radius * 0.15  # Mésoglée (couche intermédiaire)

        # Système nerveux
        self.nerve_net = DiffuseNerveNet(
            n_neurons=100,  # Nombre de neurones dans le réseau diffus
            radius=radius
        )

        # Système gastrovasculaire
        self.gastric_cavity_radius = radius * 0.6
        self.tentacles = self._init_tentacles()

        # État physiologique
        self.energy = 1.0
        self.health = 1.0

    def _init_tentacles(self, n_tentacles: int = 8) -> List['Tentacle']:
        """Initialise les tentacules autour de la méduse"""
        tentacles = []
        for i in range(n_tentacles):
            angle = (2 * np.pi * i) / n_tentacles
            base_x = self.x + self.radius * np.cos(angle)
            base_y = self.y + self.radius * np.sin(angle)
            tentacles.append(Tentacle(base_x, base_y, angle))
        return tentacles

    def update(self, dt: float):
        """Mise à jour de l'état physiologique et comportemental"""
        # Mise à jour du réseau nerveux
        neural_response = self.nerve_net.update(dt)

        # Contraction de la cloche basée sur l'activité neurale
        self.contraction_state = neural_response.mean()
        
        # Mise à jour des tentacules
        for tentacle in self.tentacles:
            tentacle.update(dt, self.contraction_state)

        # Métabolisme de base
        self.energy -= 0.01 * dt
        if self.energy < 0.2:
            self.health -= 0.05 * dt

        # Limites physiologiques
        self.energy = np.clip(self.energy, 0, 1)
        self.health = np.clip(self.health, 0, 1)