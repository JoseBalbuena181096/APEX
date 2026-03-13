# Lanzar Cámara + Lane Detection + Web Server

## Desde la PC (WSL), todo por SSH remoto a la Jetson

### 1. Matar procesos zombie anteriores

```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 "killall -9 ros2 python3 ascamera_node camera_app usb_cam_node_exe 2>/dev/null; echo 'Procesos eliminados'"
```

### 2. Recargar driver de video USB (por si la cámara quedó bloqueada)

```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 "echo 'yahboom' | sudo -S rmmod uvcvideo 2>/dev/null; echo 'yahboom' | sudo -S modprobe uvcvideo; echo 'Driver recargado'"
```

### 3. Lanzar la cámara de profundidad (ascamera)

```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 "source /opt/ros/humble/setup.bash && source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash && ros2 launch yahboomcar_depth camera_app.launch.py"
```

> Este comando se queda corriendo. Abre **otra terminal** para el siguiente paso.

### 4. Esperar ~5 segundos y lanzar lane detection + web server

```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 "source /opt/ros/humble/setup.bash && source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash && ros2 launch lane_detection lane_detection_launch.py"
```

> Este launch file lanza ambos procesos:
> - `image_subscriber` — nodo C++ de detección de carriles
> - `web_video_server.py` — servidor MJPEG en puerto 8081

### 5. Ver el resultado

Abrir en el navegador de Windows:

**http://192.168.1.168:8081/**

Se mostrará el video con la detección de carriles y sliders para ajustar parámetros en tiempo real.

## Notas

- La cámara publica en `/ascamera_hp60c/camera_publisher/rgb0/image` a ~18 FPS
- Lane detection procesa a ~7 FPS en la Jetson
- Si el video no aparece, verificar que no haya nodos duplicados con `ros2 node list`
- Los parámetros se guardan automáticamente en `~/.config/apex/lane_detection_params.json` en la Jetson
