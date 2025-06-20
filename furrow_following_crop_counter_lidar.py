import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import cv2


class SimpleFurrowFollower(Node):
    def __init__(self):
        super().__init__("simple_furrow_follower")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriber
        self.lidar_sub = self.create_subscription(
            PointCloud2, 
            "/robot/top_3d_laser/points_filtered", 
            self.lidar_callback, 
            qos_profile
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_image_pub = self.create_publisher(Image, "/lidar_debug_image", 10)
        
        self.bridge = CvBridge()

        # Parámetros OPTIMIZADOS PARA SEGUIMIENTO DE LÍNEA
        self.linear_speed = 0.4
        self.Kp = 2.0  # Control más agresivo para seguir la línea
        
        # Región de búsqueda
        self.search_width = 0.8
        self.forward_distance = 2.0
        self.min_x_distance = 0.3

        # PARÁMETROS DEL CORREDOR CENTRAL - ESTRECHO
        self.center_corridor_width = 0.25  # ±25cm del centro
        self.center_deadzone = 0.05   # Zona muerta de ±5cm
        
        # Parámetros de detección de surco
        self.min_height_percentile = 50  # Usar puntos desde el 50% más alto (menos restrictivo)
        self.min_points_per_segment = 2  # Reducir requisito mínimo de puntos
        self.adaptive_height_detection = True  # Nueva opción para detección adaptativa

        self.get_logger().info("Simple Furrow Follower started - following precise center line")

    def find_center_line_in_corridor(self, points):
        """Encuentra la línea central SOLO usando puntos dentro del corredor central"""
        if len(points) == 0:
            return None, []
        
        # Filtrar región de interés básica
        x = points[:, 0]
        y = points[:, 1] 
        z = points[:, 2]
        
        # Exclusión del robot
        robot_exclusion_radius = 0.5
        distance_from_robot = np.sqrt(x**2 + y**2)
        
        mask = (
            (x > self.min_x_distance) & 
            (x < self.forward_distance) & 
            (np.abs(y) < self.search_width) &
            (z > -0.5) &
            (distance_from_robot > robot_exclusion_radius)
        )
        
        roi_points = points[mask]
        
        if len(roi_points) < 10:
            self.get_logger().warn(f"Pocos puntos en ROI: {len(roi_points)}")
            return None, []
        
        # FILTRO CRÍTICO: Solo puntos dentro del corredor central
        corridor_mask = np.abs(roi_points[:, 1]) <= self.center_corridor_width
        corridor_points = roi_points[corridor_mask]
        
        if len(corridor_points) < 5:
            self.get_logger().warn(f"Pocos puntos en corredor central: {len(corridor_points)}")
            return None, []
        
        # DETECCIÓN ADAPTATIVA DE ALTURA
        if self.adaptive_height_detection:
            z_corridor = corridor_points[:, 2]
            z_range = np.max(z_corridor) - np.min(z_corridor)
            
            if z_range > 0.2:  # Hay variación significativa de altura
                # Usar percentil alto para surcos con plantas
                z_threshold = np.percentile(z_corridor, self.min_height_percentile)
                self.get_logger().info(f"Modo plantas: z_range={z_range:.2f}m, threshold={z_threshold:.2f}m")
            else:
                # Usar percentil más bajo para surcos sin plantas
                z_threshold = np.percentile(z_corridor, 30)  # Solo excluir el 30% más bajo
                self.get_logger().info(f"Modo sin plantas: z_range={z_range:.2f}m, threshold={z_threshold:.2f}m")
        else:
            z_threshold = np.percentile(corridor_points[:, 2], self.min_height_percentile)
        
        high_corridor_points = corridor_points[corridor_points[:, 2] >= z_threshold]
        
        if len(high_corridor_points) < 3:
            # Si no hay suficientes puntos altos, usar todos los del corredor
            self.get_logger().warn(f"Pocos puntos altos ({len(high_corridor_points)}), usando todos los del corredor")
            high_corridor_points = corridor_points
        
        # Dividir en segmentos por distancia X
        x_coords = high_corridor_points[:, 0]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        
        # Crear segmentos adaptativos según la cantidad de puntos
        if len(high_corridor_points) > 20:
            num_segments = 4
        elif len(high_corridor_points) > 10:
            num_segments = 3
        else:
            num_segments = 2
        
        segment_size = (x_max - x_min) / num_segments
        
        line_points = []
        
        for i in range(num_segments):
            segment_start = x_min + i * segment_size
            segment_end = x_min + (i + 1) * segment_size
            
            segment_mask = (x_coords >= segment_start) & (x_coords <= segment_end)
            segment_points = high_corridor_points[segment_mask]
            
            if len(segment_points) >= self.min_points_per_segment:
                # Centro de masa simple si hay pocos puntos, ponderado si hay muchos
                if len(segment_points) < 5:
                    center_x = np.mean(segment_points[:, 0])
                    center_y = np.mean(segment_points[:, 1])
                else:
                    # Centro de masa ponderado por altura
                    weights = segment_points[:, 2] - np.min(segment_points[:, 2]) + 0.1
                    center_x = np.average(segment_points[:, 0], weights=weights)
                    center_y = np.average(segment_points[:, 1], weights=weights)
                
                line_points.append([center_x, center_y])
                
                self.get_logger().info(f"Segmento {i}: X=[{segment_start:.2f}-{segment_end:.2f}], "
                                     f"Points={len(segment_points)}, Center=({center_x:.2f}, {center_y:.3f})")
        
        if len(line_points) < 1:
            self.get_logger().warn("No se pudieron crear puntos de línea")
            return None, []
        
        # Si solo hay un punto, usar su posición Y directamente
        if len(line_points) == 1:
            target_y = line_points[0][1]
            self.get_logger().info(f"Línea con 1 punto, target_y={target_y:.3f}m")
        else:
            # Ordenar puntos por distancia X
            line_points = sorted(line_points, key=lambda p: p[0])
            
            # Calcular objetivo como promedio ponderado (más peso a puntos lejanos)
            weights = [p[0] for p in line_points]  # Peso por distancia X
            target_y = np.average([p[1] for p in line_points], weights=weights)
            
            self.get_logger().info(f"Línea creada con {len(line_points)} puntos, target_y={target_y:.3f}m")
        
        return target_y, line_points

    def create_precise_debug_image(self, points, target_y, line_points):
        """Crea imagen de debug mostrando solo la línea del corredor central"""
        debug_image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        if len(points) == 0:
            cv2.putText(debug_image, "NO POINTS", (200, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return debug_image
        
        # Filtros para visualización
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        robot_exclusion_radius = 0.5
        distance_from_robot = np.sqrt(x**2 + y**2)
        
        mask = (
            (x > self.min_x_distance) &
            (x < self.forward_distance) & 
            (np.abs(y) < self.search_width) &
            (z > -0.5) &
            (distance_from_robot > robot_exclusion_radius)
        )
        
        roi_points = points[mask]
        
        # Parámetros de conversión
        scale = 100
        center_x = 300
        center_y = 350
        
        if len(roi_points) > 0:
            z_min = np.min(roi_points[:, 2])
            z_max = np.max(roi_points[:, 2])
            z_range = z_max - z_min
            
            # Dibujar todos los puntos ROI con transparencia
            for point in roi_points:
                px = int(center_x - point[1] * scale)
                py = int(center_y - point[0] * scale)
                
                if 0 <= px < 600 and 0 <= py < 400:
                    # Color basado en si está en corredor central
                    in_corridor = abs(point[1]) <= self.center_corridor_width
                    
                    if in_corridor:
                        # Puntos del corredor en verde brillante
                        if z_range > 0.001:
                            intensity = int(((point[2] - z_min) / z_range) * 255)
                        else:
                            intensity = 128
                        color = (0, intensity, 0)  # Verde variable
                        cv2.circle(debug_image, (px, py), 3, color, -1)
                    else:
                        # Puntos fuera del corredor en azul tenue
                        cv2.circle(debug_image, (px, py), 1, (100, 50, 0), -1)
        
        # Dibujar corredor central (líneas blancas)
        corridor_left = int(center_x - self.center_corridor_width * scale)
        corridor_right = int(center_x + self.center_corridor_width * scale)
        cv2.line(debug_image, (corridor_left, 0), (corridor_left, 400), (255, 255, 255), 2)
        cv2.line(debug_image, (corridor_right, 0), (corridor_right, 400), (255, 255, 255), 2)
        
        # DIBUJAR LÍNEA MORADA DEL SURCO (solo puntos del corredor)
        if len(line_points) >= 2:
            # Convertir puntos de línea a píxeles
            line_pixels = []
            for point in line_points:
                px = int(center_x - point[1] * scale)
                py = int(center_y - point[0] * scale)
                if 0 <= px < 600 and 0 <= py < 400:
                    line_pixels.append((px, py))
            
            # Dibujar línea continua morada gruesa
            if len(line_pixels) >= 2:
                for i in range(len(line_pixels) - 1):
                    cv2.line(debug_image, line_pixels[i], line_pixels[i + 1], (255, 0, 255), 6)
                
                # Círculos en cada punto de la línea
                for pixel in line_pixels:
                    cv2.circle(debug_image, pixel, 8, (255, 255, 0), -1)
                
                # Extensión hacia el horizonte
                if len(line_pixels) >= 2:
                    last_point = line_pixels[-1]
                    second_last = line_pixels[-2]
                    
                    dx = last_point[0] - second_last[0]
                    dy = last_point[1] - second_last[1]
                    
                    if dy != 0:
                        extension_length = 150
                        future_x = int(last_point[0] + dx * extension_length / abs(dy))
                        future_y = max(0, last_point[1] - extension_length)
                        
                        cv2.line(debug_image, last_point, (future_x, future_y), (255, 0, 255), 3, cv2.LINE_AA)
        
        # Dibujar robot
        cv2.circle(debug_image, (center_x, center_y), 12, (255, 255, 255), 3)
        cv2.putText(debug_image, "ROBOT", (center_x - 25, center_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Línea central de referencia
        cv2.line(debug_image, (center_x, 0), (center_x, 400), (128, 128, 128), 1)
        
        # LÍNEA ROJA DE OBJETIVO/DESVIACIÓN
        if target_y is not None:
            target_px = int(center_x - target_y * scale)
            if 0 <= target_px < 600:
                cv2.line(debug_image, (target_px, 0), (target_px, 400), (0, 0, 255), 4)
                
                # Flecha indicando dirección de corrección
                arrow_start = (target_px, 200)
                arrow_end_x = center_x if target_y > 0 else center_x
                arrow_end = (arrow_end_x, 200)
                
                cv2.arrowedLine(debug_image, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)
                
                cv2.putText(debug_image, f"TARGET: {target_y:.3f}m", (target_px + 10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Información de texto
        cv2.putText(debug_image, f"Corridor: ±{self.center_corridor_width*100:.0f}cm", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(debug_image, f"Line points: {len(line_points)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        if target_y is not None:
            error = abs(target_y)
            status = "CENTRADO" if error < self.center_deadzone else "CORRIGIENDO"
            color = (0, 255, 0) if error < self.center_deadzone else (0, 255, 255)
            cv2.putText(debug_image, f"Status: {status}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Leyenda
        cv2.putText(debug_image, "GREEN=Corridor | MAGENTA=Furrow | RED=Target", (10, 380), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return debug_image

    def lidar_callback(self, msg: PointCloud2):
        try:
            points = np.array([
                [p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            ])
            
            if len(points) == 0:
                self.get_logger().warn("Point cloud vacío")
                return
                
        except Exception as e:
            self.get_logger().error(f"Error procesando point cloud: {e}")
            return

        # Encontrar línea central del surco en el corredor
        target_y, line_points = self.find_center_line_in_corridor(points)
        
        # Crear y publicar imagen de debug
        debug_image = self.create_precise_debug_image(points, target_y, line_points)
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Error publicando debug image: {e}")
        
        # Control del robot basado en la línea detectada
        if target_y is None:
            self.get_logger().warn("No se detectó línea del surco, parando")
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            return
        
        # Control con zona muerta más precisa
        if abs(target_y) < self.center_deadzone:
            error = 0.0
            self.get_logger().info("✓ SIGUIENDO LÍNEA CENTRAL")
        else:
            error = -target_y
            direction = "←" if target_y > 0 else "→"
            self.get_logger().info(f"Corrigiendo {direction} para seguir línea")
        
        # Velocidad adaptativa según el error
        speed_factor = max(0.6, 1.0 - abs(error) * 2.0)
        
        twist = Twist()
        twist.linear.x = self.linear_speed * speed_factor
        twist.angular.z = np.clip(self.Kp * error, -1.5, 1.5)
        
        self.cmd_vel_pub.publish(twist)
        
        self.get_logger().info(
            f"Line points: {len(line_points)} | Target Y: {target_y:.3f}m | Error: {error:.3f}m | "
            f"Linear: {twist.linear.x:.2f}m/s | Angular: {twist.angular.z:.3f}rad/s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleFurrowFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
