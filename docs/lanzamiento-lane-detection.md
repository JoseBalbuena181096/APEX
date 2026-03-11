# Lanzamiento de Lane Detection + Web Server

Guia para lanzar el sistema de deteccion de carriles en la Jetson Orin Nano (Yahboom ROSMASTER A1) y visualizarlo desde un navegador en Windows.

## Conexion SSH

```bash
ssh jetson@192.168.1.168
# password: yahboom
```

O con `sshpass` para automatizar:

```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 "COMANDO"
```

## Workspaces importantes

La Jetson tiene **dos workspaces** que deben cargarse (source) para que todo funcione:

```bash
# 1. ROS 2 base
source /opt/ros/humble/setup.bash

# 2. Drivers externos (ascamera, sllidar, etc.)
source /home/jetson/yahboomcar_ros2_ws/software/library_ws/install/setup.bash

# 3. Paquetes de aplicacion (lane_detection, object_detection, etc.)
source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash
```

**IMPORTANTE**: Si no cargas `library_ws`, el paquete `ascamera` no se encuentra y la camara no arranca.

## 1. Reinicio por Software de la Camara

Si la camara no arranca, da error `uvc_open:Busy`, o la imagen sale corrupta:

### Matar procesos zombie

```bash
killall -9 ros2 python3 image_subscriber ascamera_node camera_app v4l2_camera_node usb_cam_node_exe
```

### Reiniciar el driver USB de video

```bash
sudo rmmod uvcvideo
sudo modprobe uvcvideo
```

Esto recarga el modulo del kernel sin reiniciar la Jetson. Si persiste, desconecta y reconecta fisicamente la camara USB.

### Secuencia completa de reinicio

```bash
killall -9 ros2 python3 image_subscriber ascamera_node camera_app v4l2_camera_node usb_cam_node_exe
sleep 2
sudo rmmod uvcvideo
sleep 1
sudo modprobe uvcvideo
sleep 2
# Ahora si, relanzar la camara
```

## 2. Lanzar la Camara (ascamera HP60C)

La camara del robot es una **Orbbec HP60C** (camara de profundidad). Requiere el driver `ascamera` del workspace `library_ws`. **No usar** `v4l2_camera` ni `usb_cam` directamente porque producen imagen corrupta (bandas verdes/rosa) debido a que el stream raw mezcla datos RGB + profundidad en resoluciones no estandar (640x642, 1280x1040).

### Terminal 1: Camara

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/yahboomcar_ros2_ws/software/library_ws/install/setup.bash
source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash

ros2 launch yahboomcar_depth camera_app.launch.py
```

Esto publica en el topico `/ascamera_hp60c/camera_publisher/rgb0/image` con encoding correcto a 640x480.

## 3. Lanzar Lane Detection

El nodo de lane detection se suscribe automaticamente a `/ascamera_hp60c/camera_publisher/rgb0/image` (no necesita remap si se usa `ascamera`).

### Terminal 2: Nodo de lane detection

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/yahboomcar_ros2_ws/software/library_ws/install/setup.bash
source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash

ros2 run lane_detection image_subscriber --ros-args \
  -p enable_web_view:=true \
  -p headless:=true
```

### Terminal 3: Web video server

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/yahboomcar_ros2_ws/software/library_ws/install/setup.bash
source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash

python3 /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/lane_detection/share/lane_detection/scripts/web_video_server.py
```

## 4. Ver el Resultado

Abrir en el navegador de Windows:

```
http://192.168.1.168:8081/
```

La interfaz web muestra:
- Stream MJPEG de la deteccion de carriles (imagen procesada)
- Panel de tuning con sliders para ajustar parametros en tiempo real

## Lanzamiento Rapido (Todo desde una PC con sshpass)

Ejecutar cada comando en una terminal separada:

```bash
SOURCES="source /opt/ros/humble/setup.bash && source /home/jetson/yahboomcar_ros2_ws/software/library_ws/install/setup.bash && source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash"
SSH="sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168"

# Terminal 1: Camara
$SSH "$SOURCES && ros2 launch yahboomcar_depth camera_app.launch.py"

# Terminal 2: Lane detection
$SSH "$SOURCES && ros2 run lane_detection image_subscriber --ros-args -p enable_web_view:=true -p headless:=true"

# Terminal 3: Web server
$SSH "$SOURCES && python3 /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/lane_detection/share/lane_detection/scripts/web_video_server.py"
```

## Troubleshooting

| Problema | Solucion |
|----------|----------|
| `uvc_open:Busy` | Matar procesos zombie + `sudo rmmod uvcvideo && sudo modprobe uvcvideo` |
| Imagen corrupta (bandas verdes/rosa) | Usar `ascamera` (NO `v4l2_camera` ni `usb_cam`). La camara Orbbec HP60C necesita su driver propio |
| `package 'ascamera' not found` | Falta hacer source de `library_ws`: `source /home/jetson/yahboomcar_ros2_ws/software/library_ws/install/setup.bash` |
| `Failed to send AVPacket` | No usar `usb_cam` con `mjpeg2rgb` |
| No carga imagen en web | Verificar que `web_video_server.py` este corriendo y el topico publique: `ros2 topic hz /lane_detection/debug_image` |
| `stopStreaming` inmediato en ascamera | Reiniciar driver uvcvideo y relanzar |

## Detener Todo

```bash
killall -9 image_subscriber ascamera_node python3
```
