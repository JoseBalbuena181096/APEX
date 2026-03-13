# Mejoras del Controlador — APEX Master.cpp

> Migración de PD reactivo sobre crosstrack crudo a controlador predictivo con CTE + heading + curvatura.
> Compatible con las señales nuevas de `image_subscriber.cpp` (mejora 10 del documento de detección).

---

## Estado actual del controlador

Tu PD actual hace esto:

```
steer = kp * crosstrack_error + kd * (crosstrack_error - error_last)
```

Funciona, pero tiene 4 problemas concretos que se ven en pista:

**1. La señal de entrada es píxeles crudos.** `crosstrack_error` viene en píxeles de la imagen (ej: -35, +42). Si cambias la resolución de procesamiento o el ancho de la pista, Kp y Kd dejan de servir y hay que recalibrar. No hay normalización.

**2. El derivativo amplifica ruido.** `crosstrack_error - error_last` es una derivada numérica sobre una señal ruidosa (un solo valor entero que salta frame a frame). Eso genera jitter en el steering. Por eso Kd tiene que ser muy bajo, y entonces no amortigua bien.

**3. No hay anticipación.** El controlador solo reacciona al error presente. Si viene una curva, no lo sabe hasta que el crosstrack error crece. Para entonces ya se desplazó, corrige tarde, se pasa, oscila.

**4. Velocidad constante.** `base_speed_` es fijo. En curvas debería frenar, en rectas podría ir más rápido. Sin información de curvatura no puede decidir.

---

## Controlador nuevo: 3 señales, 4 ganancias

### Señales de entrada (publicadas por `image_subscriber.cpp`)

El nodo de detección publica un `geometry_msgs::msg::Vector3` en `/lane_detection/control_ref`:

| Campo | Señal | Rango | Significado |
|-------|-------|-------|-------------|
| `x` | CTE | [-1, +1] | Error lateral normalizado. 0 = centrado, ±1 = en el borde del carril |
| `y` | Heading error | [-1, +1] | Error angular. 0 = alineado con el carril, ±1 = ±45° torcido |
| `z` | Curvatura | [-1, +1] | Curvatura del carril adelante. 0 = recto, ±1 = curva máxima |

Las 3 señales ya vienen normalizadas a [-1, +1]. El controlador no necesita saber nada de píxeles, resolución, ni ancho de carril.

### Ley de control

```
steering = Kp_cte * CTE + Kp_heading * heading + Kff * curvatura
speed    = max_speed * (1 - Kv_curve * |curvatura|)
```

Cada término resuelve un problema específico:

- **Kp_cte × CTE**: "Estoy desplazado del centro, vuelve." Corrección lateral pura.
- **Kp_heading × heading**: "Estoy torcido respecto al carril, alinéate." Esto actúa como amortiguamiento natural — frena la corrección lateral antes de que se pase.
- **Kff × curvatura**: "Viene una curva, pre-gira." Feedforward puro — no espera a que aparezca error.
- **Kv_curve × |curvatura|**: Reduce velocidad proporcionalmente a la curvatura.

### Implementación completa

```cpp
#include <cmath>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>

class Master : public rclcpp::Node {
private:
    // Subscribers
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr ref_sub_;
    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr lane_state_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr confidence_sub_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

    // Estado
    float cte_ = 0.0f;
    float heading_ = 0.0f;
    float curvature_ = 0.0f;
    float confidence_ = 0.0f;
    uint8_t lane_state_ = 0;  // 0=BOTH, 1=ONE, 2=NONE, 3=RECOVERING
    bool enabled_ = false;

    // === 4 ganancias principales ===
    double kp_cte_;        // Corrección lateral
    double kp_heading_;    // Alineación angular
    double kff_;           // Feedforward de curvatura
    double max_speed_;     // Velocidad máxima en recta

    // === Ganancia secundaria (velocidad en curvas) ===
    double kv_curve_;      // Reducción de velocidad por curvatura (0-1)

    // === Límite de steering ===
    double max_angular_;

public:
    Master() : Node("master_control") {
        // Ganancias principales
        this->declare_parameter("kp_cte", 0.5);
        this->declare_parameter("kp_heading", 0.3);
        this->declare_parameter("kff", 0.2);
        this->declare_parameter("max_speed", 0.20);

        // Secundarias
        this->declare_parameter("kv_curve", 0.5);
        this->declare_parameter("max_angular", 0.8);
        this->declare_parameter("enabled", false);

        load_params();

        param_cb_handle_ = this->add_on_set_parameters_callback(
            std::bind(&Master::on_param_change, this, std::placeholders::_1));

        // Suscripciones a las señales nuevas de image_subscriber
        ref_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/lane_detection/control_ref", 1,
            [this](const geometry_msgs::msg::Vector3::SharedPtr msg) {
                cte_ = msg->x;
                heading_ = msg->y;
                curvature_ = msg->z;
                control_loop();
            });

        lane_state_sub_ = this->create_subscription<std_msgs::msg::UInt8>(
            "/lane_detection/lane_state", 1,
            [this](const std_msgs::msg::UInt8::SharedPtr msg) {
                lane_state_ = msg->data;
            });

        confidence_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/lane_detection/confidence", 1,
            [this](const std_msgs::msg::Float32::SharedPtr msg) {
                confidence_ = msg->data;
            });

        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        RCLCPP_INFO(this->get_logger(),
            "Master v2: kp_cte=%.2f kp_head=%.2f kff=%.2f speed=%.2f",
            kp_cte_, kp_heading_, kff_, max_speed_);
    }

private:
    void load_params() {
        kp_cte_ = this->get_parameter("kp_cte").as_double();
        kp_heading_ = this->get_parameter("kp_heading").as_double();
        kff_ = this->get_parameter("kff").as_double();
        max_speed_ = this->get_parameter("max_speed").as_double();
        kv_curve_ = this->get_parameter("kv_curve").as_double();
        max_angular_ = this->get_parameter("max_angular").as_double();
        enabled_ = this->get_parameter("enabled").as_bool();
    }

    rcl_interfaces::msg::SetParametersResult on_param_change(
            const std::vector<rclcpp::Parameter> &params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &p : params) {
            const auto &name = p.get_name();
            if (name == "kp_cte") kp_cte_ = p.as_double();
            else if (name == "kp_heading") kp_heading_ = p.as_double();
            else if (name == "kff") kff_ = p.as_double();
            else if (name == "max_speed") max_speed_ = p.as_double();
            else if (name == "kv_curve") kv_curve_ = p.as_double();
            else if (name == "max_angular") max_angular_ = p.as_double();
            else if (name == "enabled") {
                enabled_ = p.as_bool();
                if (!enabled_) publish_stop();
            }
            RCLCPP_INFO(this->get_logger(), "Param '%s' = %s",
                        name.c_str(), p.value_to_string().c_str());
        }
        return result;
    }

    void publish_stop() {
        geometry_msgs::msg::Twist stop;
        cmd_vel_pub_->publish(stop);
    }

    // ============================================================
    // Loop de control — se ejecuta cada vez que llega control_ref
    // ============================================================
    void control_loop() {
        if (!enabled_) {
            publish_stop();
            return;
        }

        // ----------------------------------------------------------
        // Steering: 3 términos paralelos
        // ----------------------------------------------------------
        double steer = kp_cte_ * cte_
                     + kp_heading_ * heading_
                     + kff_ * curvature_;

        // ----------------------------------------------------------
        // Modular por confianza de detección
        //
        // Confianza baja = no confiar en la corrección fuerte.
        // Escalar steering hacia 0 (ir recto) cuando confianza baja.
        // Nunca escalar por debajo de 0.3 para que no ignore todo.
        // ----------------------------------------------------------
        float conf_scale = std::max(0.3f, confidence_);
        steer *= conf_scale;

        // ----------------------------------------------------------
        // Modular por estado de líneas
        //
        // NO_LINES: el nodo de detección ya maneja la navegación
        // inercial y publica ángulos suavizados. Aquí solo
        // reducimos la agresividad del steering.
        // ----------------------------------------------------------
        switch (lane_state_) {
            case 0: break;                       // BOTH_LINES: normal
            case 1: steer *= 0.85; break;        // ONE_LINE: un poco menos agresivo
            case 2: steer *= 0.5; break;         // NO_LINES: muy conservador
            case 3: steer *= 0.7; break;         // RECOVERING: transición
        }

        steer = std::clamp(steer, -max_angular_, max_angular_);

        // ----------------------------------------------------------
        // Velocidad: reducir en curvas, reducir sin líneas
        // ----------------------------------------------------------
        double speed = max_speed_;

        // Factor de curvatura: más curva = más lento
        speed *= (1.0 - kv_curve_ * std::abs(curvature_));

        // Factor de estado: sin líneas = más lento
        if (lane_state_ == 2) speed *= 0.6;       // NO_LINES
        else if (lane_state_ == 3) speed *= 0.8;  // RECOVERING

        // Mínimo para no quedarse parado
        speed = std::max(0.05, speed);

        // ----------------------------------------------------------
        // Publicar
        // ----------------------------------------------------------
        geometry_msgs::msg::Twist twist;
        twist.linear.x = speed;
        twist.angular.z = steer;
        cmd_vel_pub_->publish(twist);

        // Log cada 30 frames
        static int log_cnt = 0;
        if (++log_cnt % 30 == 0) {
            RCLCPP_INFO(this->get_logger(),
                "[CTRL] cte=%.2f head=%.2f curv=%.2f -> az=%.3f vx=%.3f "
                "conf=%.1f state=%d",
                cte_, heading_, curvature_, steer, speed,
                confidence_, lane_state_);
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Master>());
    rclcpp::shutdown();
    return 0;
}
```

---

## Comparación directa

| Aspecto | Master.cpp actual | Master.cpp nuevo |
|---------|-------------------|------------------|
| **Entrada** | `crosstrack_error` (Int16, píxeles) | CTE + heading + curvatura (Vector3, normalizados [-1,1]) |
| **Ley de control** | `kp*e + kd*(e - e_prev)` | `kp_cte*CTE + kp_head*heading + kff*curvatura` |
| **Derivativa** | Numérica sobre señal ruidosa | No necesaria — heading error actúa como amortiguamiento natural |
| **Anticipación** | Ninguna | Feedforward de curvatura pre-gira antes de la curva |
| **Velocidad** | Constante (`base_speed_`) | Adaptativa: reduce en curvas y sin líneas |
| **Estado de líneas** | E-stop binario (>20 frames = parar) | Modulación gradual (BOTH → ONE → NO_LINES → RECOVERING) |
| **Confianza** | No usa | Escala el steering proporcionalmente |
| **Ganancias a calibrar** | 4 (kp, kd, base_speed, max_angular) | 4+1 (kp_cte, kp_heading, kff, max_speed + kv_curve) |
| **Calibración** | kp y kd se afectan mutuamente | Cada ganancia tiene efecto independiente y visible |

---

## Guía de calibración

La calibración se hace secuencialmente. Cada paso tiene un síntoma visible y una sola perilla.

### Paso 1: Kp_cte (empezar aquí)

Poner `kp_heading = 0`, `kff = 0`, velocidad baja.

| Síntoma | Acción |
|---------|--------|
| Robot no se centra, se queda desplazado | Subir kp_cte |
| Robot oscila a los lados (zigzaguea) | Bajar kp_cte |
| Valor inicial sugerido | 0.4 - 0.6 |

### Paso 2: Kp_heading (anti-zigzag)

Con kp_cte fijo del paso anterior, subir kp_heading.

| Síntoma | Acción |
|---------|--------|
| Sigue zigzagueando con kp_cte bien | Subir kp_heading |
| Robot corrige tan suave que no termina de alinearse | Subir kp_heading |
| Robot reacciona exagerado a cambios de dirección del carril | Bajar kp_heading |
| Valor inicial sugerido | 0.2 - 0.4 |

**Por qué esto reemplaza Kd**: El heading error mide hacia dónde apunta el robot respecto al carril. Si el robot se acerca al centro pero apunta hacia él (va a pasarse), el heading error genera un steering opuesto que frena la corrección antes del overshoot. Es el mismo efecto que un derivativo, pero sobre una señal limpia en vez de una derivada numérica ruidosa.

### Paso 3: Kff (anticipación en curvas)

Llevar el robot a una curva conocida.

| Síntoma | Acción |
|---------|--------|
| Robot entra tarde a la curva, se desplaza hacia afuera | Subir kff |
| Robot gira antes de tiempo, se mete al interior | Bajar kff |
| Valor inicial sugerido | 0.1 - 0.3 |

### Paso 4: max_speed y Kv_curve

Con el steering calibrado, subir velocidad.

| Síntoma | Acción |
|---------|--------|
| Robot va lento pero estable | Subir max_speed |
| Robot se sale en curvas a velocidad alta | Subir kv_curve (frena más en curvas) |
| Robot casi se detiene en curvas | Bajar kv_curve |
| Valores iniciales sugeridos | max_speed = 0.20, kv_curve = 0.4-0.6 |

---

## Qué se eliminó y por qué

**E-stop binario eliminado.** Tu código actual para el robot cuando no ve líneas por 20 frames. Eso no sirve cuando las líneas se quitan intencionalmente. Ahora el nodo de detección maneja la navegación inercial (mejora 8) y el controlador solo reduce la agresividad del steering y la velocidad proporcionalmente al estado de las líneas. El robot nunca frena completamente por falta de líneas — solo se vuelve más conservador.

**Kd eliminado.** La derivada numérica sobre `crosstrack_error` era la fuente de jitter. El heading error hace el mismo trabajo (amortiguar) pero sobre una señal que ya viene suavizada del nodo de detección (EMA temporal sobre los polinomios). Si después de calibrar Kp_cte y Kp_heading sientes que falta amortiguamiento, puedes agregar un `Kd_cte * (cte_ - prev_cte_)` como quinto parámetro, pero probablemente no lo necesites.

**`distance_center` y `angle` eliminados como entrada.** Se reemplazan por las 3 señales normalizadas. Durante la transición, el nodo de detección publica ambos formatos (viejo y nuevo) en paralelo, así que puedes correr ambos controladores y comparar con rosbag.

---

## Migración gradual

No necesitas reemplazar todo de golpe:

**Fase 1 — Solo observar**: El nodo de detección publica las señales nuevas en `/lane_detection/control_ref` en paralelo con las viejas. El controlador actual sigue funcionando. Grabas un rosbag y comparas las señales.

**Fase 2 — Probar el nuevo controlador**: Lanzas el Master nuevo en paralelo (remapeando su `/cmd_vel` a un topic dummy). Comparas el steering que calcula cada uno sobre los mismos datos del rosbag.

**Fase 3 — Cambiar**: Cuando el nuevo genera mejor steering en el rosbag, lo conectas al `/cmd_vel` real y calibras las 4 ganancias en pista.