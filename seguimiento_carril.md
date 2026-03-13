# Seguimiento de Carril — Procedimiento de Lanzamiento

## Requisitos
- Robot Yahboom ROSMASTER-A1 con Jetson Nano
- Conexión SSH: `jetson@192.168.1.168` (password: `yahboom`)
- Desde PC con WSL: necesita `sshpass` instalado

## Procedimiento Completo

### Paso 1: Matar todos los procesos anteriores
```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "killall -9 ros2 python3 ascamera_node camera_app usb_cam_node_exe image_subscriber master Ackman_driver_A1 2>/dev/null; sleep 3; echo 'Procesos eliminados'"
```

### Paso 2: Recargar driver USB de la cámara
```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "echo 'yahboom' | sudo -S rmmod uvcvideo 2>/dev/null; sleep 2; echo 'yahboom' | sudo -S modprobe uvcvideo; sleep 2; echo 'Driver recargado'"
```
> **Importante**: Esperar 2 segundos entre rmmod y modprobe para que el kernel libere el dispositivo.

### Paso 3: Lanzar cámara de profundidad
```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "nohup bash -c 'source /opt/ros/humble/setup.bash && source /home/jetson/yahboomcar_ros2_ws/software/library_ws/install/setup.bash && source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash && ros2 launch yahboomcar_depth camera_app.launch.py' > /tmp/camera.log 2>&1 &"
```
> **Nota**: Se DEBEN sourcear 3 workspaces: ros humble + software/library_ws (ascamera) + yahboomcar_ws.

### Paso 4: Esperar ~8 segundos y verificar cámara
```bash
sleep 8
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "source /opt/ros/humble/setup.bash && ros2 topic list 2>/dev/null | grep rgb"
```
Debe mostrar:
```
/ascamera_hp60c/camera_publisher/rgb0/camera_info
/ascamera_hp60c/camera_publisher/rgb0/image
```

### Paso 5: Lanzar sistema de navegación (lane detection + controlador + web servers)
```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "nohup bash -c 'source /opt/ros/humble/setup.bash && source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash && ros2 launch navigation_control navigation_without_obstacles_launch.py' > /tmp/navigation.log 2>&1 &"
```

### Paso 6: Lanzar driver del chasis (motores + servo)
```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "nohup bash -c 'source /opt/ros/humble/setup.bash && source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash && ros2 run yahboomcar_bringup Ackman_driver_A1' > /tmp/driver.log 2>&1 &"
```

### Paso 7: Verificar todos los nodos
```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "source /opt/ros/humble/setup.bash && ros2 node list"
```
Debe mostrar:
```
/ascamera_hp60c/camera_publisher
/control_tuning_server
/driver_node
/image_compressor
/joy_node
/line_detection
/master_control
/web_video_server
```

### Paso 8: Activar el controlador
Ir a `http://192.168.1.168:8082` y presionar **START**, o por terminal:
```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "source /opt/ros/humble/setup.bash && ros2 param set /master_control enabled true"
```

## Interfaces Web
- **Vista de cámara + debug**: `http://192.168.1.168:8081`
- **Tuning del controlador**: `http://192.168.1.168:8082`

## Nodos del Sistema

| Nodo | Función |
|------|---------|
| `/ascamera_hp60c/camera_publisher` | Cámara de profundidad HP60C |
| `/line_detection` | Detección de carriles (bird-eye + RANSAC) |
| `/master_control` | Controlador predictivo (CTE + heading + curvatura) |
| `/web_video_server` | Stream MJPEG debug en puerto 8081 |
| `/control_tuning_server` | Web UI de tuning en puerto 8082 |
| `/driver_node` | Driver del chasis Ackermann (motores + servo) |

## Parámetros Calibrados

### Controlador (Master)
| Parámetro | Valor | Función |
|-----------|-------|---------|
| kp_cte | 0.7 | Corrección lateral (centrado) |
| kp_heading | 0.5 | Alineación angular (seguir curvas) |
| kff | 0.45 | Feedforward de curvatura (anticipar curvas) |
| max_speed | 0.30 | Velocidad máxima en recta (m/s) |
| kv_curve | 0.5 | Reducción de velocidad en curvas |
| max_angular | 0.8 | Límite de steering |

### Trapecio (Perspectiva)
| Parámetro | Valor |
|-----------|-------|
| persp_top_left_pct | 20 |
| persp_top_right_pct | 80 |
| persp_top_y_pct | 60 |
| persp_bot_left_pct | 0 |
| persp_bot_right_pct | 100 |
| persp_bot_y_pct | 75 |

## Troubleshooting

### La cámara no envía imagen
1. Matar todo (Paso 1)
2. Recargar driver USB (Paso 2) — esperar los 2 segundos
3. Relanzar desde Paso 3

### El robot no se mueve
- Verificar que `/driver_node` esté en `ros2 node list`
- Verificar que `enabled=true`: `ros2 param get /master_control enabled`
- Verificar `cmd_vel`: `ros2 topic echo /cmd_vel --once`

### FPS bajo (<15)
- Verificar que no hay otros procesos pesados en la Jetson
- Normal: 25-28 FPS a 320x240

## Compilar cambios y desplegar
```bash
# Copiar archivo modificado
sshpass -p 'yahboom' scp -o StrictHostKeyChecking=no \
  /home/jose/APEX/yahboomcar_ros2_ws/yahboomcar_ws/src/lane_detection/src/image_subscriber.cpp \
  jetson@192.168.1.168:/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/lane_detection/src/

# Compilar en el robot
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "cd /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select lane_detection"

# Para navigation_control:
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 \
  "cd /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select navigation_control"
```
