import math
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pygame
import colorsys
import numpy as np
from src.config.settings import render_config

@dataclass
class VisualizationConfig:
    """Configuration du système de rendu"""
    # Dimensions de la fenêtre
    window_width: int = 1200
    window_height: int = 800
    target_fps: int = 60
    
    # Couleurs (R,G,B)
    background_color: Tuple[int, int, int] = (10, 20, 40)
    epidermis_color: Tuple[int, int, int] = (200, 220, 255)
    mesoglea_color: Tuple[int, int, int] = (180, 200, 240)
    nerve_color: Tuple[int, int, int] = (255, 100, 100)
    
    # Options de rendu
    enable_antialiasing: bool = True
    enable_vsync: bool = True
    show_debug: bool = False
    draw_nerve_net: bool = True  # Ajout de cette option
    
    # Paramètres de rendu
    base_body_scale: float = 2.0
    min_render_size: float = 40.0
    max_render_size: float = 400.0
    
    # Effets visuels
    transparency_enabled: bool = True
    glow_effect: bool = True
    water_effect: bool = True
    
    # Interface
    show_stats: bool = True
    show_controls: bool = True
    show_motion_trails: bool = True

class CnidarianRenderer:
    """Système de rendu pour la visualisation des cnidaires"""
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        pygame.init()
        
        # Création de la fenêtre
        self.screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height)
        )
        pygame.display.set_caption("Cnidarian Simulation")
        
        # Ajout du Clock pour le FPS
        self.clock = pygame.time.Clock()
        self.current_fps = 0
        
        # Surfaces de rendu
        self.main_surface = pygame.Surface(
            (self.config.window_width, self.config.window_height),
            pygame.SRCALPHA
        )
        self.effect_surface = pygame.Surface(
            (self.config.window_width, self.config.window_height),
            pygame.SRCALPHA
        )
        
        # Police pour le texte
        self.font = pygame.font.Font(None, 24)
        
        # Effets d'eau
        self.water_offset = 0
        self.water_points = self._generate_water_points()
        
        # Cache pour les gradients et textures
        self.gradient_cache = {}
        self.texture_cache = {}
        
        # Pré-calcul des surfaces fréquemment utilisées
        self.cached_surfaces = {}
        self._init_cached_surfaces()
        
        # Système de cache
        self.surface_cache = {}
        self._init_surface_cache()
        self.motion_trails = {}
        
    def _init_surface_cache(self):
        """Pré-calcule les surfaces fréquemment utilisées"""
        # Cache pour différentes tailles de corps
        for size in range(10, 101, 5):
            key = f"body_{size}"
            surface = self._create_body_surface(size)
            self.surface_cache[key] = surface
            
        # Cache pour les effets de brillance
        glow_surface = pygame.Surface((100, 100), pygame.SRCALPHA)
        for r in range(50, 0, -1):
            alpha = int(200 * (r / 50))
            pygame.draw.circle(glow_surface, (255, 255, 255, alpha), (50, 50), r)
        self.surface_cache["glow"] = glow_surface

    def _get_cached_surface(self, key: str, size: int) -> pygame.Surface:
        """Récupère une surface du cache, la plus proche de la taille demandée"""
        cache_key = f"{key}_{round(size/5)*5}"
        if cache_key not in self.surface_cache:
            # Création et mise en cache si nécessaire
            self.surface_cache[cache_key] = self._create_body_surface(size)
        return self.surface_cache[cache_key]        
        
    def _init_cached_surfaces(self):
        """Pré-calcule les surfaces fréquemment utilisées"""
        # Pré-calcul des dégradés pour différentes tailles
        for size in range(10, 101, 5):  # De 10 à 100 par pas de 5
            surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            
            # Dégradé de base pour cette taille
            for r in range(size, 0, -1):
                alpha = int(255 * (r / size) * 0.7)
                pygame.draw.circle(
                    surface,
                    (*self.config.mesoglea_color, alpha),
                    (size, size),
                    r
                )
            
            self.cached_surfaces[size] = surface        
        
    def _generate_water_points(self) -> List[Tuple[float, float]]:
        """Génère des points pour l'effet d'eau"""
        points = []
        for x in range(0, self.config.window_width + 20, 20):
            for y in range(0, self.config.window_height + 20, 20):
                points.append((x + np.random.normal(0, 2), 
                             y + np.random.normal(0, 2)))
        return points
        
    def render(self, cnidarians: List['CnidarianOrganism'], 
               environment_data: Optional[Dict] = None):
        """Rendu principal optimisé"""
        # Nettoyage des surfaces
        self.main_surface.fill((0, 0, 0, 0))
        self.effect_surface.fill((0, 0, 0, 0))
        
        # Rendu de l'arrière-plan
        self._render_background(environment_data)

        if environment_data:
            self._render_prey(environment_data)
            self._render_current_field(environment_data)
        
        # Rendu groupé des cnidaires
        for cnidarian in sorted(cnidarians, key=lambda c: c.y):  # Tri par profondeur
            self._render_cnidarian(cnidarian)
        
        # Application des effets en une seule fois
        if self.config.water_effect:
            self._render_water_effect()

        # Préparation de l'overlay avant le blit final
        self.current_fps = self.clock.get_fps()
        if self.config.show_stats:
            self._render_stats(cnidarians)
        if self.config.show_controls:
            self._render_controls()
        
        # Application finale
        self.screen.blit(self.main_surface, (0, 0))
        self.screen.blit(self.effect_surface, (0, 0))
        
        pygame.display.flip()
        self.clock.tick(self.config.target_fps)
            
    def _render_background(self, environment_data: Optional[Dict]):
        """Rendu du fond marin"""
        # Gradient de base
        gradient = self._get_gradient(
            self.config.background_color,
            (4, 12, 26),
            self.config.window_height
        )
        self.screen.blit(gradient, (0, 0))

        # Halo diffus pour donner une profondeur visuelle
        bloom_surface = pygame.Surface(
            (self.config.window_width, self.config.window_height),
            pygame.SRCALPHA
        )
        for idx, center_x in enumerate((180, 620, 1040)):
            radius = 220 + idx * 40
            pygame.draw.circle(
                bloom_surface,
                (30, 70, 120, 20),
                (center_x, 120 + idx * 40),
                radius
            )
        self.screen.blit(bloom_surface, (0, 0))

        # Lignes de caustiques douces
        current_time = time.time()
        for x in range(-100, self.config.window_width + 120, 110):
            points = []
            for y in range(0, self.config.window_height + 40, 30):
                wave = 18 * np.sin(current_time * 0.9 + y * 0.015 + x * 0.01)
                points.append((x + wave, y))
            if len(points) > 1:
                pygame.draw.lines(self.screen, (120, 180, 220, 18), False, points, 2)
        
        # Particules en suspension si données environnementales disponibles
        if environment_data and 'particle_density' in environment_data:
            self._render_particles(environment_data['particle_density'])
            
    def _render_water_effect(self):
        """Rendu de l'effet d'eau avec particules"""
        current_time = time.time()
        
        # Déplacement des particules
        for i, point in enumerate(self.water_points):
            # Mouvement sinusoïdal
            offset_x = 5 * np.sin(current_time + point[1] * 0.1)
            offset_y = 5 * np.cos(current_time + point[0] * 0.1)
            
            # Position avec offset
            pos = (int(point[0] + offset_x), int(point[1] + offset_y))
            
            # Opacité variable
            alpha = int(100 * (0.5 + 0.5 * np.sin(current_time * 2 + i * 0.1)))
            
            # Rendu de la particule
            pygame.draw.circle(self.effect_surface, (255, 255, 255, alpha), pos, 1)
            
    def _render_cnidarian(self, cnidarian: 'CnidarianOrganism'):
        """Rendu complet du cnidaire"""
        if self.config.show_motion_trails:
            self._render_motion_trail(cnidarian)

        # 1. Corps principal avec dégradé
        self._render_body(cnidarian)
        
        # 2. Réseau nerveux si activé 
        if self.config.draw_nerve_net:
            self._render_nerve_net(cnidarian)
            
        # 3. Tentacules
        self._render_tentacles(cnidarian)
        
        # 4. Effets visuels d'activité
        if cnidarian.is_active:
            self._render_activity_glow(cnidarian)
        self._render_velocity_indicator(cnidarian)
            
    def _create_body_surface(self, size: int) -> pygame.Surface:
        """Crée une surface de base pour le corps d'un cnidaire d'une taille donnée"""
        # Surface avec canal alpha
        surface = pygame.Surface((size * 2 + 10, size * 2 + 10), pygame.SRCALPHA)
        center = (size + 5, size + 5)
        
        # 1. Mésoglée (couche interne) avec dégradé radial
        for r in range(size, 0, -1):
            opacity = int(255 * (r / size) * 0.7)
            color = (*self.config.mesoglea_color, opacity)
            pygame.draw.circle(surface, color, center, r)

        # 2. Épiderme (couche externe) plus claire
        edge_color = tuple(min(255, c + 40) for c in self.config.mesoglea_color)
        pygame.draw.circle(surface, (*edge_color, 180), center, size, 2)
        
        # 3. Effet de volume avec un reflet
        highlight_pos = (center[0] - size//3, center[1] - size//3)
        highlight_radius = size // 4
        for r in range(highlight_radius, 0, -1):
            opacity = int(100 * (r / highlight_radius))
            pygame.draw.circle(surface, (255, 255, 255, opacity), highlight_pos, r)
        
        return surface

    def _render_body(self, cnidarian: 'CnidarianOrganism'):
        """Rendu amélioré du corps vu de dessus"""
        # Calcul de la taille de rendu
        render_size = cnidarian.growth_system.size * self.config.base_body_scale
        render_size = max(self.config.min_render_size, 
                         min(self.config.max_render_size, render_size))
        
        # Création de la surface du corps
        size = int(cnidarian.radius)
        body_surface = pygame.Surface((size * 2 + 10, size * 2 + 10), pygame.SRCALPHA)
        center = (size + 5, size + 5)
        pulse = getattr(cnidarian, 'body_pulse', 0.0)
        swim_stretch_x = 1.0 + 0.08 * pulse
        swim_stretch_y = 1.0 - 0.12 * pulse
        speed = float(np.linalg.norm(cnidarian.velocity))
        motion_tilt = min(0.18, speed * 0.03) * np.sin(cnidarian.orientation)
        
        # 1. Ombre portée légère
        shadow_radius = size + 2
        shadow_color = (0, 0, 20, 50)
        shadow_rect = pygame.Rect(0, 0, int(shadow_radius * 2.1 * swim_stretch_x), int(shadow_radius * 2.0 * swim_stretch_y))
        shadow_rect.center = (center[0] + 2, center[1] + 2)
        pygame.draw.ellipse(body_surface, shadow_color, shadow_rect)
        
        # 2. Corps principal avec dégradé radial (ombrelle)
        for r in range(size, 0, -1):
            opacity = int(255 * (r / size) * 0.8)
            color = (*self.config.mesoglea_color, opacity)
            ellipse_rect = pygame.Rect(0, 0, max(2, int(r * 2 * swim_stretch_x)), max(2, int(r * 2 * swim_stretch_y)))
            ellipse_rect.center = center
            pygame.draw.ellipse(body_surface, color, ellipse_rect)
            
        # 3. Cavité gastrique (plus foncée et profonde)
        gastric_radius = int(size * (0.35 + 0.03 * pulse))
        for r in range(gastric_radius, 0, -1):
            darkness = int(150 * (1 - r/gastric_radius))
            color = (
                max(0, self.config.mesoglea_color[0] - darkness),
                max(0, self.config.mesoglea_color[1] - darkness),
                max(0, self.config.mesoglea_color[2] - darkness),
                200
            )
            ellipse_rect = pygame.Rect(0, 0, max(2, int(r * 1.8 * swim_stretch_x)), max(2, int(r * 1.7 * swim_stretch_y)))
            ellipse_rect.center = center
            pygame.draw.ellipse(body_surface, color, ellipse_rect)
            
        # 4. Point central plus foncé (bouche)
        mouth_radius = int(size * 0.1)
        pygame.draw.circle(body_surface, (10, 20, 40, 255), center, mouth_radius)
        
        # 5. Effet de brillance sur l'ombrelle
        highlight_pos = (center[0] - size//3, center[1] - size//3)
        highlight_radius = size // 3
        for r in range(highlight_radius, 0, -1):
            opacity = int(100 * (r / highlight_radius))
            pygame.draw.circle(body_surface, (255, 255, 255, opacity), highlight_pos, r)
        
        # 6. Stries radiales subtiles
        n_stries = 16
        for i in range(n_stries):
            angle = (2 * np.pi * i) / n_stries
            start_pos = (
                center[0] + int(size * 0.4 * np.cos(angle)),
                center[1] + int(size * 0.4 * np.sin(angle))
            )
            end_pos = (
                center[0] + int(size * 0.9 * swim_stretch_x * np.cos(angle + motion_tilt)),
                center[1] + int(size * 0.9 * swim_stretch_y * np.sin(angle))
            )
            pygame.draw.line(body_surface, (255, 255, 255, 30), start_pos, end_pos, 1)

        # 7. Anneau contractile visible pendant la nage
        if pulse > 0.05:
            ring_rect = pygame.Rect(0, 0, int(size * 1.65 * swim_stretch_x), int(size * 1.45 * swim_stretch_y))
            ring_rect.center = center
            pygame.draw.ellipse(body_surface, (220, 240, 255, int(45 + 70 * pulse)), ring_rect, 2)
        
        # Application sur la surface principale
        screen_pos = (int(cnidarian.x - size - 5), int(cnidarian.y - size - 5))
        self.main_surface.blit(body_surface, screen_pos)

    def _render_motion_trail(self, cnidarian: 'CnidarianOrganism'):
        """Affiche une traînée discrète pour rendre le déplacement lisible."""
        trail = self.motion_trails.setdefault(id(cnidarian), [])
        trail.append((float(cnidarian.x), float(cnidarian.y)))
        if len(trail) > 18:
            trail.pop(0)

        if len(trail) < 2:
            return

        for index in range(1, len(trail)):
            start = trail[index - 1]
            end = trail[index]
            alpha = int(18 + 70 * (index / len(trail)))
            width = max(1, int(cnidarian.radius * 0.08 * (index / len(trail))))
            pygame.draw.line(
                self.effect_surface,
                (120, 210, 255, alpha),
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                width
            )

    def _render_velocity_indicator(self, cnidarian: 'CnidarianOrganism'):
        """Affiche un sillage et un cap quand l'organisme bouge."""
        speed = float(np.linalg.norm(cnidarian.velocity))
        if speed < 0.05:
            return

        direction = cnidarian.velocity / max(speed, 1e-6)
        start = np.array([cnidarian.x, cnidarian.y], dtype=float)
        end = start + direction * min(55.0, 12.0 + speed * 12.0)
        side = np.array([-direction[1], direction[0]], dtype=float)

        pygame.draw.line(
            self.effect_surface,
            (140, 230, 255, 160),
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            2
        )
        tip_left = end - direction * 8 + side * 4
        tip_right = end - direction * 8 - side * 4
        pygame.draw.polygon(
            self.effect_surface,
            (180, 240, 255, 140),
            [(int(end[0]), int(end[1])), (int(tip_left[0]), int(tip_left[1])), (int(tip_right[0]), int(tip_right[1]))]
        )

        pulse = getattr(cnidarian, 'body_pulse', 0.0)
        if pulse > 0.08:
            wake_center = start - direction * (cnidarian.radius * (0.7 + pulse * 0.4))
            wake_radius = int(cnidarian.radius * (0.35 + pulse * 0.2))
            pygame.draw.circle(
                self.effect_surface,
                (120, 210, 255, int(40 + pulse * 80)),
                (int(wake_center[0]), int(wake_center[1])),
                max(2, wake_radius),
                2
            )
    
    def _render_nerve_net(self, cnidarian: 'CnidarianOrganism'):
        """Rendu du réseau nerveux"""
        if not cnidarian.nerve_net:
            return
            
        nerve_surface = pygame.Surface((int(cnidarian.radius*2.2), int(cnidarian.radius*2.2)), pygame.SRCALPHA)
        
        # Rendu des neurones et connexions
        for neuron in cnidarian.nerve_net.neurons:
            # Position relative
            rel_pos = (neuron.x + cnidarian.radius*1.1, neuron.y + cnidarian.radius*1.1)
            
            # Connexions avec effet de pulse pour les neurones actifs
            for connection in neuron.connections:
                target_neuron = connection[0]
                weight = connection[1]
                target_pos = (target_neuron.x + cnidarian.radius*1.1, 
                            target_neuron.y + cnidarian.radius*1.1)
                
                # Couleur basée sur l'activité
                if neuron.is_active or target_neuron.is_active:
                    connection_color = (255, 100, 100, 150)  # Rouge pour actif
                    # Effet de pulse sur l'épaisseur
                    pulse = 1 + 0.5 * np.sin(time.time() * 10)
                    width = int(1 * pulse)
                else:
                    connection_color = (150, 150, 200, 100)  # Bleu pour inactif
                    width = 1
                
                # Dessin de la connexion
                pygame.draw.line(nerve_surface, connection_color, rel_pos, target_pos, width)
            
            # Dessin du neurone
            neuron_radius = 2
            if neuron.is_active:
                neuron_color = (255, 100, 100, 200)  # Rouge vif pour actif
                neuron_radius = 3
            else:
                neuron_color = (150, 150, 200, 150)  # Bleu pour inactif
                
            pygame.draw.circle(nerve_surface, neuron_color, 
                             (int(rel_pos[0]), int(rel_pos[1])), 
                             neuron_radius)
        
        # Application de la surface du réseau nerveux
        pos = (int(cnidarian.x - cnidarian.radius*1.1),
               int(cnidarian.y - cnidarian.radius*1.1))
        self.main_surface.blit(nerve_surface, pos)
        
    def _render_tentacles(self, cnidarian: 'CnidarianOrganism'):
        """Rendu amélioré des tentacules vus de dessus"""
        for tentacle in cnidarian.tentacles:
            # Surface pour le tentacule
            surface_size = int(tentacle.params.base_length * 2)
            tentacle_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            
            # Point d'attache sur l'ombrelle
            base_pos = (surface_size//2, surface_size//2)
            
            # Création d'un motif en spirale pour le tentacule
            points = []
            max_radius = tentacle.params.base_length * 0.8
            n_points = 20
            
            for i in range(n_points):
                t = i / (n_points - 1)
                # Spirale logarithmique
                radius = max_radius * (1 - np.exp(-2 * t))
                angle = tentacle.base_angle + t * np.pi * 0.5
                
                # Ajout d'ondulation
                wave = np.sin(time.time() * 3 + t * 4) * (t * 5)
                radius += wave
                
                x = base_pos[0] + radius * np.cos(angle)
                y = base_pos[1] + radius * np.sin(angle)
                points.append((x, y))
            
            # Dessin du tentacule avec dégradé
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]
                
                # Largeur décroissante
                width = max(1, int(6 * (1 - i/len(points))))
                
                # Couleur avec transparence croissante
                alpha = int(200 * (1 - i/len(points)))
                color = (*self.config.epidermis_color, alpha)
                
                # Dessin du segment
                pygame.draw.line(tentacle_surface, color, start, end, width)
                
                # Ajout de cnidocytes
                if width > 1 and i % 2 == 0:
                    self._render_cnidocyte(tentacle_surface, start, alpha)
            
            # Application avec offset basé sur la pulsation
            pulse = np.sin(time.time() * 2 + tentacle.base_angle) * 2
            pos = (
                int(cnidarian.x - surface_size//2 + pulse),
                int(cnidarian.y - surface_size//2 + pulse)
            )
            self.main_surface.blit(tentacle_surface, pos)
            
    def _calculate_tentacle_curve(self, tentacle: 'Tentacle') -> List[Tuple[float, float]]:
        """Calcule une courbe de Bézier pour le tentacule"""
        points = []
        segments = tentacle.segments
        
        # Point de base du tentacule
        points.append((
            tentacle.params.base_length*1.5,
            tentacle.params.base_length*1.5
        ))
        
        # Points de contrôle intermédiaires
        for i, segment in enumerate(segments):
            t = i / (len(segments) - 1)
            
            # Position de base
            x = tentacle.params.base_length * 1.5
            y = tentacle.params.base_length * (1.5 + t)
            
            # Ajout d'ondulation
            wave = np.sin(time.time() * 2 + t * 4) * (t * 20)
            x += wave
            
            # Effet de courbure naturelle
            curve = t * t * 30
            x += curve
            
            points.append((x, y))
        
        return self._smooth_curve(points)            
           
    def _smooth_curve(self, points: List[Tuple[float, float]], steps: int = 10) -> List[Tuple[float, float]]:
        """Lisse une courbe en utilisant l'interpolation de Catmull-Rom"""
        if len(points) < 4:
            return points
            
        smooth_points = []
        
        # Duplication des points aux extrémités pour la continuité
        p = [points[0]] + points + [points[-1]]
        
        for i in range(1, len(p) - 2):
            for t in range(steps):
                # Paramètre d'interpolation
                t = t / steps
                
                # Points de contrôle
                p0, p1, p2, p3 = p[i-1:i+3]
                
                # Interpolation Catmull-Rom
                x = 0.5 * (
                    (2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t**2 +
                    (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t**3
                )
                
                y = 0.5 * (
                    (2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t**2 +
                    (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t**3
                )
                
                smooth_points.append((x, y))
                
        return smooth_points

    def _render_cnidocyte(self, surface: pygame.Surface, 
                         position: Tuple[float, float], 
                         base_alpha: int):
        """Rendu d'un cnidocyte individuel"""
        glow_radius = 2 + np.sin(time.time() * 5) * 0.5
        
        # Effet de brillance
        for r in range(3, 0, -1):
            alpha = int(base_alpha * (r/3))
            pygame.draw.circle(
                surface,
                (255, 255, 255, alpha),
                (int(position[0]), int(position[1])),
                int(glow_radius * r/2)
            )
      
    def _render_cnidocytes(self, surface: pygame.Surface, 
                          segment: 'TentacleSegment',
                          start: Tuple[float, float],
                          end: Tuple[float, float]):
        """Rendu amélioré des cnidocytes"""
        for cnidocyte in segment.cnidocytes:
            if not cnidocyte.is_charged:
                continue
                
            # Position relative sur le segment
            t = np.random.random()  # Position aléatoire le long du segment
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            # Effet de brillance pulsante
            glow_radius = 2 + np.sin(time.time() * 5) * 0.5
            glow_intensity = int(200 * (0.7 + 0.3 * np.sin(time.time() * 8)))
            
            # Couches de brillance
            for r in range(int(glow_radius * 3), 0, -1):
                alpha = int(glow_intensity * (r / (glow_radius * 3)))
                pygame.draw.circle(
                    surface,
                    (255, 255, 255, alpha),
                    (int(x), int(y)),
                    r
                )
                
            # Point central plus brillant
            pygame.draw.circle(
                surface,
                (255, 255, 255, glow_intensity),
                (int(x), int(y)),
                1
            )
      
    def _render_water_environment(self):
        """Rendu des effets d'eau ambiants"""
        self.water_offset = (self.water_offset + 0.02) % (2 * np.pi)
        
        # Particules en suspension
        for point in self.water_points:
            x = point[0] + np.sin(self.water_offset + point[1]/50) * 2
            y = point[1] + np.cos(self.water_offset + point[0]/50) * 2
            
            # Variation d'opacité
            alpha = int(30 + 20 * np.sin(self.water_offset + x/100))
            
            pygame.draw.circle(
                self.effect_surface,
                (255, 255, 255, alpha),
                (int(x), int(y)),
                1
            )

    def _render_prey(self, environment_data: Dict):
        """Rend les proies visibles comme points lumineux mobiles."""
        prey_items = environment_data.get('prey', [])
        for prey in prey_items:
            x, y = prey['position']
            size = max(2, int(prey['parameters']['size'] * 0.25))
            glow = pygame.Surface((size * 6, size * 6), pygame.SRCALPHA)
            center = (glow.get_width() // 2, glow.get_height() // 2)
            for radius in range(size * 2, 0, -1):
                alpha = int(80 * (radius / max(1, size * 2)))
                pygame.draw.circle(glow, (120, 220, 255, alpha), center, radius)
            self.effect_surface.blit(glow, (int(x - center[0]), int(y - center[1])))
            pygame.draw.circle(self.main_surface, (180, 245, 255, 220), (int(x), int(y)), size)

    def _render_current_field(self, environment_data: Dict):
        """Rend quelques vecteurs de courant pour matérialiser la dérive de l'eau."""
        flow_samples = environment_data.get('flow_samples', [])
        for sample in flow_samples:
            x, y = sample['position']
            flow_x, flow_y = sample['flow']
            magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
            if magnitude < 0.05:
                continue

            scale = 14.0
            end = (x + flow_x * scale, y + flow_y * scale)
            pygame.draw.line(
                self.effect_surface,
                (90, 150, 210, 70),
                (int(x), int(y)),
                (int(end[0]), int(end[1])),
                1
            )

    def _render_activity_glow(self, cnidarian: 'CnidarianOrganism'):
        """Rendu de l'effet de brillance lors de l'activité"""
        if not cnidarian.is_active:
            return
            
        glow_surface = pygame.Surface(
            (int(cnidarian.radius*3), int(cnidarian.radius*3)),
            pygame.SRCALPHA
        )
        
        # Pulsation de la brillance
        intensity = 0.5 + 0.5 * np.sin(time.time() * 4)
        
        # Gradient radial de brillance
        center = (glow_surface.get_width()//2, glow_surface.get_height()//2)
        max_radius = int(cnidarian.radius * 1.5)
        
        for r in range(max_radius, 0, -1):
            alpha = int(50 * (r/max_radius) * intensity)
            color = (*self.config.epidermis_color, alpha)
            pygame.draw.circle(glow_surface, color, center, r)
        
        # Application de la brillance
        pos = (
            int(cnidarian.x - glow_surface.get_width()//2),
            int(cnidarian.y - glow_surface.get_height()//2)
        )
        self.effect_surface.blit(glow_surface, pos)            
            
    def _render_activity_effects(self, cnidarian: 'CnidarianOrganism'):
        """Rendu des effets visuels liés à l'activité"""
        # Effet de pulsation
        pulse_size = cnidarian.radius * (1 + 0.1 * np.sin(time.time() * 4))
        
        # Couleur basée sur l'état
        if cnidarian.is_feeding:
            effect_color = (255, 200, 100, 50)  # Orange pour l'alimentation
        elif cnidarian.is_swimming:
            effect_color = (100, 200, 255, 50)  # Bleu pour la nage
        else:
            effect_color = (200, 200, 255, 30)  # Blanc pour l'activité générale
            
        pygame.draw.circle(
            self.effect_surface,
            effect_color,
            (int(cnidarian.x), int(cnidarian.y)),
            int(pulse_size)
        )
        
    def _get_gradient(self, color1: Tuple[int, int, int],
                     color2: Tuple[int, int, int],
                     height: int) -> pygame.Surface:
        """Crée un gradient vertical entre deux couleurs"""
        key = (color1, color2, height)
        if key in self.gradient_cache:
            return self.gradient_cache[key]
            
        surface = pygame.Surface((1, height))
        for y in range(height):
            ratio = y / height
            color = self._blend_colors(color1, color2, ratio)
            surface.set_at((0, y), color)
            
        gradient = pygame.transform.scale(
            surface,
            (self.config.window_width, height)
        )
        self.gradient_cache[key] = gradient
        return gradient
        
    def _render_stats(self, cnidarians: List['CnidarianOrganism']):
        """Affichage des statistiques"""
        stats = [
            f"Cnidaires: {len(cnidarians)}",
            f"FPS: {int(self.current_fps)}",  # Utilisation du FPS stocké
            f"Temps: {time.strftime('%H:%M:%S')}"
        ]
        
        # Statistiques globales
        if cnidarians:
            avg_energy = sum(c.energy for c in cnidarians) / len(cnidarians)
            avg_health = sum(c.health for c in cnidarians) / len(cnidarians)
            stats.extend([
                f"Énergie moyenne: {avg_energy:.2f}",
                f"Santé moyenne: {avg_health:.2f}"
            ])
        
        # Rendu du texte
        y_offset = 10
        for stat in stats:
            text_surface = self.font.render(stat, True, (255, 255, 255))
            self.main_surface.blit(text_surface, (10, y_offset))
            y_offset += 25
            
    def _render_controls(self):
        """Affichage des contrôles"""
        controls = [
            "ESC: Quitter",
            "ESPACE: Pause",
            "R: Réinitialiser",
            "N: Afficher/Masquer réseau nerveux",
            "D: Mode debug",
            "+/-: Zoom"
        ]
        
        y_offset = self.config.window_height - (len(controls) * 25 + 10)
        for control in controls:
            text_surface = self.font.render(control, True, (200, 200, 200))
            self.main_surface.blit(
                text_surface, 
                (self.config.window_width - text_surface.get_width() - 10, y_offset)
            )
            y_offset += 25
            
    def _render_particles(self, density: float):
        """Rendu des particules en suspension dans l'eau"""
        n_particles = int(density * 1000)
        for _ in range(n_particles):
            x = np.random.randint(0, self.config.window_width)
            y = np.random.randint(0, self.config.window_height)
            size = np.random.randint(1, 3)
            alpha = np.random.randint(20, 60)
            
            pygame.draw.circle(
                self.effect_surface,
                (255, 255, 255, alpha),
                (x, y),
                size
            )
            
    def _apply_glow_effect(self):
        """Applique un effet de lueur"""
        # Création d'une copie floue de la surface principale
        blur_surface = pygame.Surface(
            (self.config.window_width, self.config.window_height),
            pygame.SRCALPHA
        )
        blur_surface.blit(self.main_surface, (0, 0))
        
        # Application du flou gaussien
        for _ in range(3):
            pygame.transform.gaussian_blur(blur_surface, 2)
            
        # Fusion avec transparence
        self.main_surface.blit(blur_surface, (0, 0), special_flags=pygame.BLEND_ADD)
        
    def _get_neuron_color(self, neuron: 'CnidarianNeuron') -> Tuple[int, int, int, int]:
        """Détermine la couleur d'un neurone basée sur son activité"""
        if neuron.is_active:
            # Couleur plus vive pour les neurones actifs
            base_color = (255, 100, 100)
            alpha = 255
        else:
            # Couleur plus douce pour les neurones au repos
            base_color = (200, 150, 150)
            alpha = 150
            
        # Modulation par le potentiel membranaire
        intensity = (neuron.membrane_potential - neuron.params.resting_potential) / \
                   (neuron.params.threshold - neuron.params.resting_potential)
        intensity = max(0, min(1, intensity))
        
        return (*base_color, int(alpha * intensity))
        
    @staticmethod
    def _blend_colors(color1: Tuple[int, int, int], 
                     color2: Tuple[int, int, int],
                     ratio: float) -> Tuple[int, int, int]:
        """Mélange deux couleurs selon un ratio"""
        return tuple(
            int(c1 * (1 - ratio) + c2 * ratio)
            for c1, c2 in zip(color1, color2)
        )
        
    def cleanup(self):
        """Nettoyage des ressources"""
        pygame.quit()
