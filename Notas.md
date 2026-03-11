# Notas sobre ROS 2 y Jetson (Yahboom ROSMASTER A1)

## Cómo lanzar la cámara en la Jetson Orin Nano

Para poder visualizar los tópicos de la cámara, primero debes encenderla ejecutando su nodo correspondiente en la Jetson.

1. **Conéctate por SSH a la Jetson:**
   ```bash
   ssh jetson@192.168.1.168
   ```
   *(La contraseña es `yahboom`)*

2. **Lanza el nodo de la cámara:**
   Dependiendo del modelo de cámara que tengas instalado:

   - **Cámara de profundidad (Astra):**
     ```bash
     ros2 launch yahboomcar_depth camera_app.launch.py
     ```

   - **Cámara USB estándar:**
     ```bash
     ros2 launch usb_cam camera.launch.py
     ```

3. **Verificar los tópicos en tu PC:**
   Una vez lanzado el comando anterior en la Jetson, abre otra terminal en tu PC y verifica que los tópicos se están publicando:
   ```bash
   ```
   Deberías ver tópicos como `/ascamera_hp60c/camera_publisher/rgb0/image` o similares.

## Solución de problemas comunes (Troubleshooting)

### Error: `uvc_open:Busy` o cámara no arranca
Si al lanzar la cámara recibes errores indicando que el dispositivo está ocupado, o el proceso se queda congelado en la terminal y no publica tópicos, significa que hay procesos "zombie" de ROS 2 anteriores usando la cámara en segundo plano.

1. **Cierra los procesos colgados en la Jetson:**
   Ejecuta el siguiente comando para forzar el cierre de cualquier nodo de ROS o cámara de Yahboom que haya quedado atascado:
   ```bash
   killall -9 ros2 python3 ascamera_node camera_app usb_cam_node_exe
   ```

2. **Recarga el driver de video (opcional):**
   Si el problema persiste incluso después de matar los procesos, puedes reiniciar el módulo del kernel encargado de la cámara USB sin tener que reiniciar la Jetson entera:
   ```bash
   sudo rmmod uvcvideo
   sudo modprobe uvcvideo
   ```

3. **Vuelve a lanzar la cámara:**
   Una vez limpios los procesos, ejecuta de nuevo el comando `.launch.py` de tu cámara (paso 2 de la sección anterior).

## Ver Detección de Carril por Web (Sin RViz en Windows)

Como RViz y `rqt_image_view` suelen tener problemas para recibir el video en WSL debido a la red, creamos un servidor proxy en Python que envía la imagen procesada de la detección de líneas por HTTP.

1. **Lanza la cámara:**
   (Usando el comando `ros2 launch yahboomcar_depth camera_app.launch.py`)
2. **Lanza el nodo de detección de líneas (C++):**
   ```bash
   source ~/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash
   ros2 run lane_detection image_subscriber
   ```
3. **Lanza el proxy Web de video:**
   ```bash
   python3 /tmp/camera_web_server_lane.py
   ```
4. **Ver el resultado:**
   Abre un navegador en Windows y entra a `http://192.168.1.168:8081/`.

## Enviar comandos remotos sin entrar a SSH interactivo (`sshpass`)

Para automatizar el lanzamiento de los nodos sin tener que abrir múltiples pestañas de terminal e iniciar sesión manualmente, puedes usar `sshpass` y enviarle la instrucción completa como "un solo comando". 

El formato es el siguiente:

```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 "COMANDO_A_EJECUTAR_AQUI"
```

### Ejemplo completo: Lanzar la detección de carriles remoto

Dado que los comandos de ROS necesitan que *"sourcees"* (cargues) el entorno de ROS 2 y tu workspace en cada nueva terminal, el comando a enviar se ve así:

```bash
sshpass -p 'yahboom' ssh -o StrictHostKeyChecking=no jetson@192.168.1.168 "source /opt/ros/humble/setup.bash && source /home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash && ros2 run lane_detection image_subscriber"
```

- `-p 'yahboom'`: Contraseña de la Jetson.
- `-o StrictHostKeyChecking=no`: Evita que SSH te pregunte *Are you sure you want to continue connecting?*.
- `"source ... && ros2 run ..."`: Carga las variables de entorno de ROS de la Jetson y luego inicia el programa, todo en la misma línea usando `&&`.
