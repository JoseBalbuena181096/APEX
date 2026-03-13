# Mejoras para Lane Detection — APEX (Yahboom ROS2)

> Guía de implementación para `image_subscriber.cpp`
> Contexto: El robot llega a una pista desconocida, tiene un breve periodo de calibración, y luego conduce autónomamente sin mapa previo. En competencia pueden quitar una o ambas líneas de tramos de la pista.

---

## 1. Auto-calibración inicial + adaptación continua de iluminación

### 1A. Calibración inicial (robot detenido, 2 segundos)

**Qué mide**: Propiedades de la pista que NO cambian durante la carrera — ancho de carril y color de las líneas.

```cpp
// Nuevos miembros
enum class CalibState { CALIBRATING, RUNNING };
CalibState calib_state_ = CalibState::CALIBRATING;
int calib_frame_count_ = 0;
int calib_frames_needed_ = 60;  // ~2 seg a 30 fps

// Solo propiedades geométricas y de color (NO iluminación)
std::vector<float> calib_widths_;
float calibrated_lane_width_ = 0.0f;
int calibrated_hue_center_ = 0;
bool lines_are_white_ = true;  // vs color (amarillo, etc.)

void calibration_step(const Mat& birdeye_binary, const Mat& birdeye_hls) {
    // 1. Medir ancho entre picos del histograma
    int* peaks = Histogram(birdeye_binary);
    float width = (float)(peaks[1] - peaks[0]);
    if (width > 20 && width < PROC_WIDTH * 0.8f) {
        calib_widths_.push_back(width);
    }

    // 2. Determinar si las líneas son blancas o de color
    std::vector<Mat> channels;
    split(birdeye_hls, channels);
    int color_votes = 0, white_votes = 0;

    for (int r = PROC_HEIGHT/2; r < PROC_HEIGHT; r += 4) {  // Submuestreo
        for (int c = 0; c < PROC_WIDTH; c += 4) {
            if (birdeye_binary.at<uchar>(r, c) > 0) {
                float s = channels[2].at<uchar>(r, c);
                if (s > 40) {
                    color_votes++;
                    calibrated_hue_center_ = channels[0].at<uchar>(r, c);
                } else {
                    white_votes++;
                }
            }
        }
    }

    calib_frame_count_++;
    if (calib_frame_count_ >= calib_frames_needed_) {
        // Ancho: mediana
        if (!calib_widths_.empty()) {
            std::sort(calib_widths_.begin(), calib_widths_.end());
            calibrated_lane_width_ = calib_widths_[calib_widths_.size() / 2];
            lane_width_px_ = (int)calibrated_lane_width_;
        }
        lines_are_white_ = (white_votes > color_votes);
        calib_state_ = CalibState::RUNNING;

        RCLCPP_INFO(get_logger(),
            "CALIBRACIÓN: lane_width=%d px, lineas=%s",
            lane_width_px_, lines_are_white_ ? "blancas" : "color");
    }
}

// En LineDetectionCb:
if (calib_state_ == CalibState::CALIBRATING) {
    // Pipeline normal pero sin publicar dirección
    // ... warp + threshold ...
    calibration_step(img_edges, birdeye_hls);
    angle = 90;
    distance_center = 0;
    return;
}
```

### 1B. Adaptación continua de iluminación (cada frame, ~0 costo extra)

**Problema**: La iluminación cambia zona a zona dentro de la misma pista (sombras, focos, ventanas). Un `adaptive_c_` fijo para toda la carrera no sirve.

**Por qué NO impacta FPS**: No hacemos ningún cálculo nuevo. Reutilizamos datos que el pipeline YA genera. Solo agregamos una EMA de 3 sumas — eso son ~6 operaciones de punto flotante por frame.

```cpp
// Nuevos miembros (NO son vectores, NO hay allocación)
float illum_pixel_density_ = 0.0f;  // Densidad de píxeles blancos en bird-eye
float illum_ema_ = 0.0f;            // EMA de densidad
bool illum_has_history_ = false;
float adaptive_c_base_ = -25.0f;    // Valor base del adaptive_c

// En Threshold(), DESPUÉS de la morfología (ya tienes la imagen binaria):
// Contar píxeles blancos — esto es UN cv::countNonZero, ~0.02ms en 320x240
Mat Threshold(Mat frame) {
    // ... tu código actual de CLAHE + adaptiveThreshold + morfología ...

    // NUEVO: una sola línea para medir iluminación
    int white_count = cv::countNonZero(binary);
    update_illumination(white_count, binary.total());

    return binary;
}

// Función ultra-ligera: 6 operaciones float, sin malloc, sin bucles
void update_illumination(int white_pixels, size_t total_pixels) {
    float density = (float)white_pixels / (float)total_pixels;

    if (!illum_has_history_) {
        illum_ema_ = density;
        illum_has_history_ = true;
    } else {
        // EMA muy suave (alpha=0.1) para no reaccionar a un frame ruidoso
        illum_ema_ = 0.1f * density + 0.9f * illum_ema_;
    }

    // Adaptar adaptive_c_ según densidad de blancos
    //
    // Lógica:
    // - Densidad ALTA (>15%) = demasiado brillo, muchos falsos positivos
    //   → hacer adaptive_c_ más negativo (más exigente)
    // - Densidad BAJA (<3%) = zona oscura, estamos perdiendo líneas
    //   → hacer adaptive_c_ menos negativo (más permisivo)
    // - Densidad NORMAL (3-15%) = no tocar
    //
    // Rango: adaptive_c_ entre -40 (muy exigente) y -10 (muy permisivo)

    if (illum_ema_ > 0.15f) {
        adaptive_c_base_ = std::max(-40.0f, adaptive_c_base_ - 0.3f);
    } else if (illum_ema_ < 0.03f) {
        adaptive_c_base_ = std::min(-10.0f, adaptive_c_base_ + 0.3f);
    } else {
        adaptive_c_base_ = 0.95f * adaptive_c_base_ + 0.05f * (-25.0f);
    }

    adaptive_c_ = (int)adaptive_c_base_;
}
```

**Análisis de costo**:

| Operación | Tiempo estimado (320×240) |
|-----------|--------------------------|
| `cv::countNonZero(binary)` | ~0.02 ms |
| 6 operaciones float (EMA + comparaciones) | ~0.001 ms |
| **Total extra por frame** | **~0.02 ms** |
| Frame completo actual (~30 FPS) | ~33 ms |
| **Overhead** | **0.06%** |

**Por qué funciona**: `adaptiveThreshold` ya se adapta localmente (block=51). Pero la constante C define "cuánto más brillante que el promedio local debe ser un píxel". Cuando toda la escena es muy brillante, el promedio local sube y muchas superficies pasan el umbral → C debe ser más negativo. Cuando hay sombra, pocas cosas pasan → C se relaja. Ajustamos C muy lentamente (EMA α=0.1 + cambio de ±0.3 por frame) para que el umbral se acomode a la zona de la pista sin oscilar.

**Qué NO adapta continuamente** (y por qué):
- **Ancho de carril**: Fijo para toda la pista. Se mide en calibración inicial.
- **Color de las líneas**: No cambia con iluminación, es propiedad del material.
- **CLAHE clip limit**: Ya normaliza contraste localmente, no necesita ajuste global.
- **Block size del adaptive threshold**: Cambiarlo en runtime causa saltos bruscos.

**Visualización de debug** (opcional, solo si `enable_web_view_`):

```cpp
if (enable_web_view_) {
    int bar_w = (int)(illum_ema_ * 200);
    Scalar bar_color;
    if (illum_ema_ > 0.15f)
        bar_color = Scalar(0, 0, 255);    // Rojo: demasiado brillante
    else if (illum_ema_ < 0.03f)
        bar_color = Scalar(255, 0, 0);    // Azul: demasiado oscuro
    else
        bar_color = Scalar(0, 255, 0);    // Verde: OK

    rectangle(frame, Point(10, out_h - 30),
              Point(10 + bar_w, out_h - 20), bar_color, -1);

    char buf[64];
    snprintf(buf, sizeof(buf), "C=%d dens=%.1f%%",
             adaptive_c_, illum_ema_ * 100);
    putText(frame, buf, Point(10, out_h - 35),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
}
```

---

## 2. Corrección de distorsión de lente

**Problema**: El warp de perspectiva amplifica la distorsión del lente en los extremos del trapecio. Una línea recta puede curvarse en bird-eye y generar coeficientes cuadráticos falsos.

```cpp
// Nuevos miembros
cv::Mat camera_matrix_, dist_coeffs_;
cv::Mat map1_, map2_;
bool calibration_loaded_ = false;

// Cargar desde archivo YAML (calibración del lente, se hace UNA vez por cámara)
void load_lens_calibration(const std::string& calib_file) {
    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        RCLCPP_WARN(get_logger(), "Sin calibración de lente");
        return;
    }
    fs["camera_matrix"] >> camera_matrix_;
    fs["dist_coeffs"] >> dist_coeffs_;

    cv::initUndistortRectifyMap(
        camera_matrix_, dist_coeffs_, cv::Mat(),
        cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_,
                                       cv::Size(PROC_WIDTH, PROC_HEIGHT), 0),
        cv::Size(PROC_WIDTH, PROC_HEIGHT), CV_16SC2, map1_, map2_);
    calibration_loaded_ = true;
}

// En LineDetectionCb, ANTES de Perspective():
if (calibration_loaded_) {
    cv::remap(proc_frame, proc_frame, map1_, map2_, cv::INTER_LINEAR);
}
```

**Cómo calibrar** (se hace una sola vez, no en cada pista): Imprimir un patrón de tablero de ajedrez, tomar 15-20 fotos desde distintos ángulos con la cámara del robot, correr `cv::calibrateCamera()`, guardar en YAML.

```cpp
// calibrate_camera.cpp (offline, no es parte del nodo)
std::vector<std::vector<Point3f>> obj_points;
std::vector<std::vector<Point2f>> img_points;
Size board(9, 6);

for (auto& img_path : images) {
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    std::vector<Point2f> corners;
    if (findChessboardCorners(img, board, corners)) {
        cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                     TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.001));
        img_points.push_back(corners);
        // ... llenar obj_points con coordenadas 3D del tablero
    }
}

Mat cam_matrix, dist_coeffs;
calibrateCamera(obj_points, img_points, image_size, cam_matrix, dist_coeffs,
                noArray(), noArray());

FileStorage fs("camera_calib.yaml", FileStorage::WRITE);
fs << "camera_matrix" << cam_matrix;
fs << "dist_coeffs" << dist_coeffs;
```

---

## 3. Histograma robusto con detección de picos

**Problema**: `max_element` en cada mitad puede elegir un reflejo como pico. Si ambas líneas están en la misma mitad (curva cerrada), una no se detecta.

```cpp
struct Peak {
    int position;
    int strength;
};

std::vector<Peak> find_peaks(const std::vector<int>& histogram,
                              int min_distance, int min_height) {
    std::vector<Peak> peaks;
    int n = histogram.size();

    for (int i = 1; i < n - 1; i++) {
        if (histogram[i] > min_height &&
            histogram[i] >= histogram[i - 1] &&
            histogram[i] >= histogram[i + 1]) {

            bool is_dominant = true;
            for (auto it = peaks.begin(); it != peaks.end(); ) {
                if (std::abs(it->position - i) < min_distance) {
                    if (histogram[i] > it->strength) {
                        it = peaks.erase(it);
                    } else {
                        is_dominant = false;
                        break;
                    }
                } else {
                    ++it;
                }
            }
            if (is_dominant) {
                peaks.push_back({i, histogram[i]});
            }
        }
    }

    std::sort(peaks.begin(), peaks.end(),
              [](const Peak& a, const Peak& b) { return a.strength > b.strength; });
    return peaks;
}

// En Histogram():
int min_distance = img.cols / 4;
int min_height = 5;
auto peaks = find_peaks(smoothed, min_distance, min_height);

if (peaks.size() >= 2) {
    LanePosition[0] = std::min(peaks[0].position, peaks[1].position);
    LanePosition[1] = std::max(peaks[0].position, peaks[1].position);
} else if (peaks.size() == 1) {
    // Usar ancho calibrado para inferir el otro
    float w = (calibrated_lane_width_ > 0) ? calibrated_lane_width_ : (float)lane_width_px_;
    if (peaks[0].position < img.cols / 2) {
        LanePosition[0] = peaks[0].position;
        LanePosition[1] = peaks[0].position + (int)w;
    } else {
        LanePosition[1] = peaks[0].position;
        LanePosition[0] = peaks[0].position - (int)w;
    }
} else {
    LanePosition[0] = img.cols / 4;
    LanePosition[1] = img.cols * 3 / 4;
}
```

---

## 4. Validación de curvatura del polinomio

**Problema**: RANSAC puede producir parábolas absurdas (radio de curvatura de centímetros) con puntos ruidosos.

```cpp
// Nuevo parámetro
double max_curvature_radius_;

// Declarar:
this->declare_parameter("max_curvature_radius", 80.0);

// Después de polyfit_ransac, validar:
bool validate_curvature(float* coeffs) {
    float mid_row = PROC_HEIGHT / 2.0f;
    float second_deriv = 2.0f * coeffs[2];

    if (std::abs(second_deriv) > 1e-6f) {
        float first_deriv = coeffs[1] + 2.0f * coeffs[2] * mid_row;
        float radius = std::pow(1.0f + first_deriv * first_deriv, 1.5f)
                      / std::abs(second_deriv);
        if (radius < max_curvature_radius_) {
            RCLCPP_INFO(get_logger(),
                "Curvatura rechazada: R=%.1f px < min %.1f", radius, max_curvature_radius_);
            return false;
        }
    }
    return true;
}

// En regression_left() / regression_right(), después del RANSAC:
if (ok) {
    ok = validate_curvature(polyleft);  // o polyright
}
```

**Calibración del umbral**: Para pistas de mesa típicas (~30 cm de radio real), 60-100 px en bird-eye 320×240 es razonable.

---

## 5. Inferencia de carril adaptativa

**Problema**: El offset constante `lane_width_px_` falla en curvas porque el ancho aparente cambia.

```cpp
float width_ema_ = 0.0f;
bool has_width_history_ = false;

// En draw_lines(), cuando AMBOS carriles se detectan:
if (find_line_left && find_line_right) {
    // Medir ancho real en 3 alturas
    float total_w = 0;
    float weights[] = {0.5f, 0.3f, 0.2f};
    float rows[] = {(float)(img.rows - 1), (float)(img.rows / 2), 0.0f};

    for (int i = 0; i < 3; i++) {
        float r = rows[i];
        float lc = draw_left[0] + draw_left[1]*r + draw_left[2]*r*r;
        float rc = draw_right[0] + draw_right[1]*r + draw_right[2]*r*r;
        total_w += (rc - lc) * weights[i];
    }

    if (!has_width_history_) {
        width_ema_ = total_w;
        has_width_history_ = true;
    } else {
        width_ema_ = 0.7f * total_w + 0.3f * width_ema_;
    }
}

// Al inferir carril faltante:
float effective_width = has_width_history_ ? width_ema_
                      : (calibrated_lane_width_ > 0 ? calibrated_lane_width_
                         : (float)lane_width_px_);

// Prioridad: medición en vivo > calibración inicial > parámetro fijo
```

---

## 6. Filtro de color para reducir ruido

**Problema**: Cualquier superficie brillante pasa el umbral. Las líneas tienen un color que no se aprovecha.

```cpp
// Nuevos parámetros
int sat_max_white_;
bool use_color_filter_;

// Declarar:
this->declare_parameter("use_color_filter", true);
this->declare_parameter("sat_max_white", 60);

// En Threshold(), DESPUÉS de la morfología:
if (use_color_filter_) {
    // Extraer S del HLS que ya calculaste
    Mat S = channels[2];  // channels ya existe de split(frameHLS)

    // Para líneas BLANCAS: baja saturación = probablemente línea
    Mat white_mask;
    inRange(S, 0, sat_max_white_, white_mask);

    // Combinar: solo mantener píxeles brillantes Y poco saturados
    binary = binary & white_mask;
}
```

**Auto-calibración del umbral de saturación** (integrar con mejora 1): Durante calibración, medir la saturación media de los píxeles detectados como línea. Usar `media + 2*desviación` como `sat_max_white_`.

```cpp
// En finalize_calibration():
if (!calib_sat_values_.empty()) {
    float mean_s = 0, var_s = 0;
    for (float s : calib_sat_values_) mean_s += s;
    mean_s /= calib_sat_values_.size();
    for (float s : calib_sat_values_) var_s += (s - mean_s) * (s - mean_s);
    var_s = std::sqrt(var_s / calib_sat_values_.size());
    sat_max_white_ = (int)(mean_s + 2.0f * var_s);
    sat_max_white_ = std::clamp(sat_max_white_, 30, 120);
}
```

---

## 7. Publicación de confianza de detección

**Problema**: El controlador no sabe cuándo la detección es mala y debe ser conservador.

```cpp
rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr confidence_pub_;

// En constructor:
confidence_pub_ = this->create_publisher<std_msgs::msg::Float32>(
    "/lane_detection/confidence", 10);

// Modificar polyfit_ransac para retornar ratio de inliers:
struct RansacResult {
    bool success;
    float inlier_ratio;
    int inlier_count;
};

// Calcular confianza compuesta:
float compute_confidence(bool left_ok, bool right_ok,
                          float left_inlier_ratio, float right_inlier_ratio,
                          int left_pts, int right_pts) {
    float conf = 0.0f;

    // Factor 1: ratio de inliers (40%)
    float ir = 0.0f;
    int count = 0;
    if (left_ok)  { ir += left_inlier_ratio;  count++; }
    if (right_ok) { ir += right_inlier_ratio; count++; }
    if (count > 0) ir /= count;
    conf += ir * 0.4f;

    // Factor 2: cantidad de puntos (30%)
    float pt_score = std::min(1.0f, (left_pts + right_pts) / 400.0f);
    conf += pt_score * 0.3f;

    // Factor 3: ambos detectados, no inferidos (30%)
    conf += (left_ok && right_ok) ? 0.3f : 0.15f;

    return std::clamp(conf, 0.0f, 1.0f);
}

// Publicar en cada frame:
std_msgs::msg::Float32 conf_msg;
conf_msg.data = compute_confidence(find_line_left, find_line_right,
                                    left_result.inlier_ratio,
                                    right_result.inlier_ratio,
                                    left_points.size(), right_points.size());
confidence_pub_->publish(conf_msg);
```

**Uso en el controlador**:
- `confidence < 0.3` → frenar o velocidad mínima
- `confidence < 0.5` → reducir ganancia del PID
- `confidence > 0.8` → control normal

---

## 8. Navegación sin líneas (líneas removidas intencionalmente)

**Problema real**: En competencia pueden quitar una o ambas líneas de tramos de la pista. Esto NO es un error de detección — es una condición esperada. El robot no debe frenar ni resetear; debe seguir derecho usando la última trayectoria confiable y transicionar suavemente cuando las líneas reaparezcan.

**Escenarios**:
- **0 líneas**: Tramo completamente sin marcas. Seguir recto con la última dirección conocida.
- **1 línea**: Quitan una línea. Inferir la otra (tu código ya hace esto), pero necesitamos que sea más robusto.
- **Transición**: Las líneas reaparecen. No saltar bruscamente al nuevo polinomio.

**Implementación — máquina de estados**:

```cpp
// Nuevos miembros
enum class LaneState {
    BOTH_LINES,     // Conducción normal con ambas líneas
    ONE_LINE,       // Solo una línea visible, inferir la otra
    NO_LINES,       // Sin líneas, navegación inercial
    RECOVERING      // Líneas reapareciendo, transición suave
};
LaneState lane_state_ = LaneState::BOTH_LINES;
int frames_no_lines_ = 0;
int frames_recovering_ = 0;

// Último ángulo y distancia confiables (cuando teníamos 2 líneas)
float last_good_angle_ = 90.0f;
float last_good_distance_ = 0.0f;
// Dirección inercial: derivada del ángulo para mantener curvas
float angle_rate_ = 0.0f;       // Cambio de ángulo por frame
float prev_angle_ = 90.0f;

// Parámetros
int recovery_blend_frames_;     // Frames para transicionar al regresar
float max_inertial_frames_;     // Máx frames en navegación sin líneas
float inertial_straighten_rate_;  // Qué tan rápido vuelve a recto

// Declarar:
this->declare_parameter("recovery_blend_frames", 10);
this->declare_parameter("max_inertial_frames", 90);    // ~3 seg a 30 fps
this->declare_parameter("inertial_straighten_rate", 0.02f);
```

**Lógica principal (reemplaza la sección de draw_lines donde decides qué hacer)**:

```cpp
void update_lane_state(bool left_ok, bool right_ok) {
    LaneState prev_state = lane_state_;

    if (left_ok && right_ok) {
        if (prev_state == LaneState::NO_LINES ||
            prev_state == LaneState::ONE_LINE) {
            // Reaparecieron líneas: transicionar suave
            lane_state_ = LaneState::RECOVERING;
            frames_recovering_ = 0;
        } else if (prev_state == LaneState::RECOVERING) {
            frames_recovering_++;
            if (frames_recovering_ >= recovery_blend_frames_) {
                lane_state_ = LaneState::BOTH_LINES;
            }
        } else {
            lane_state_ = LaneState::BOTH_LINES;
        }
        frames_no_lines_ = 0;

    } else if (left_ok || right_ok) {
        lane_state_ = LaneState::ONE_LINE;
        frames_no_lines_ = 0;

    } else {
        lane_state_ = LaneState::NO_LINES;
        frames_no_lines_++;
    }
}
```

**Navegación inercial (cuando no hay líneas)**:

```cpp
void navigate_inertial(Mat &img) {
    // Seguir con el último ángulo bueno, pero gradualmente
    // enderezarse (las pistas sin líneas suelen ser rectas —
    // si fuera curva, pondrían líneas).

    float target = 90.0f;
    float blend = std::min(1.0f,
        frames_no_lines_ * inertial_straighten_rate_);
    float inertial_angle = last_good_angle_ * (1.0f - blend)
                         + target * blend;

    // Si teníamos un rate de giro, mantenerlo un poco
    // (por si entramos sin líneas en medio de una curva)
    if (frames_no_lines_ < 15) {
        inertial_angle += angle_rate_ * (15 - frames_no_lines_);
    }

    angle = (int)std::clamp(inertial_angle, 45.0f, 135.0f);
    distance_center = 0;  // Asumir centrado

    // Indicador visual
    putText(img, "INERTIAL NAV", Point(10, img.rows - 20),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 165, 255), 2);

    // Barra de "combustible inercial" que se agota
    float fuel = 1.0f - (float)frames_no_lines_ / max_inertial_frames_;
    fuel = std::max(0.0f, fuel);
    int bar_width = (int)(fuel * 100);
    rectangle(img, Point(10, img.rows - 10),
              Point(10 + bar_width, img.rows - 5),
              Scalar(0, (int)(fuel * 255), (int)((1-fuel) * 255)), -1);
}
```

**Transición suave al recuperar líneas (RECOVERING)**:

```cpp
void blend_recovery(float new_angle, float new_distance) {
    float t = (float)frames_recovering_ / recovery_blend_frames_;
    t = t * t * (3.0f - 2.0f * t);  // Smoothstep para suavidad

    angle = (int)(last_good_angle_ * (1.0f - t) + new_angle * t);
    distance_center = (int)(last_good_distance_ * (1.0f - t)
                          + new_distance * t);
}
```

**Integración en draw_lines()**:

```cpp
void draw_lines(Mat &img) {
    bool find_line_right = regression_right();
    bool find_line_left = regression_left();

    // Actualizar máquina de estados
    update_lane_state(find_line_left, find_line_right);

    // Guardar rate de ángulo ANTES de modificar angle
    angle_rate_ = (float)angle - prev_angle_;
    prev_angle_ = (float)angle;

    switch (lane_state_) {
        case LaneState::BOTH_LINES: {
            // Tu código actual de draw con ambas líneas
            // ...
            // Guardar como referencia para inercial
            last_good_angle_ = (float)angle;
            last_good_distance_ = (float)distance_center;
            break;
        }

        case LaneState::ONE_LINE: {
            // Tu código actual de inferencia (ya lo tienes)
            // left+width o right-width
            // ...
            // También actualizar last_good_ porque una línea
            // sigue siendo información válida
            last_good_angle_ = (float)angle;
            last_good_distance_ = (float)distance_center;
            break;
        }

        case LaneState::NO_LINES: {
            navigate_inertial(img);
            break;
        }

        case LaneState::RECOVERING: {
            // Calcular ángulo/distancia nuevos normalmente
            float new_angle = (float)angle;
            float new_dist = (float)distance_center;
            blend_recovery(new_angle, new_dist);
            break;
        }
    }
}
```

**Publicar el estado para el controlador**:

```cpp
// Nuevo publisher
rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr lane_state_pub_;

// En constructor:
lane_state_pub_ = this->create_publisher<std_msgs::msg::UInt8>(
    "/lane_detection/lane_state", 10);

// En cada frame:
std_msgs::msg::UInt8 state_msg;
state_msg.data = static_cast<uint8_t>(lane_state_);
lane_state_pub_->publish(state_msg);

// El controlador puede usarlo así:
// 0 = BOTH_LINES   → velocidad normal, PID normal
// 1 = ONE_LINE     → velocidad normal, PID ganancia reducida 20%
// 2 = NO_LINES     → velocidad reducida 40%, solo inercial
// 3 = RECOVERING   → velocidad reducida 20%, transicionando
```

**Ajuste de confianza según estado** (integrar con mejora 7):

```cpp
float state_confidence_factor() {
    switch (lane_state_) {
        case LaneState::BOTH_LINES:  return 1.0f;
        case LaneState::ONE_LINE:    return 0.7f;
        case LaneState::NO_LINES: {
            float fuel = 1.0f - (float)frames_no_lines_ / max_inertial_frames_;
            return std::max(0.1f, 0.5f * fuel);
        }
        case LaneState::RECOVERING: {
            float t = (float)frames_recovering_ / recovery_blend_frames_;
            return 0.5f + 0.5f * t;  // 0.5 → 1.0 durante recovery
        }
    }
    return 0.0f;
}
```

---

## 9. ROI dinámico basado en velocidad

**Problema**: El trapecio es fijo. A baja velocidad conviene ver cerca (más confiable), a alta velocidad conviene ver lejos (anticipar curvas).

```cpp
rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;
float current_speed_ = 0.0f;

// En constructor:
vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "/cmd_vel", 10,
    [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        current_speed_ = std::abs(msg->linear.x);
    });

// En LineDetectionCb, antes de configurar Source[]:
float speed_factor = std::clamp(current_speed_ / 0.5f, 0.0f, 1.0f);

// A mayor velocidad, mirar más arriba (más lejos en el mundo)
float dynamic_top_y_pct = persp_top_y_pct_ - (int)(speed_factor * 15);
dynamic_top_y_pct = std::max(40.0f, (float)dynamic_top_y_pct);

float ty = PROC_HEIGHT * dynamic_top_y_pct / 100.0f;
// El resto del setup de Source[] usa ty
```

---

## 10. Señal de control mejorada: reemplazar ángulo por CTE + curvatura

**Problema con el ángulo actual**:

1. **Mezcla dos errores distintos**: El error lateral (estoy desplazado) y el error angular (estoy torcido) se funden en un número. Un PID no puede distinguir "centrado pero girando" de "desplazado pero alineado".
2. **Pierde el signo**: Haces `if (angle < 0) angle = -angle`. El controlador no sabe si girar a la izquierda o derecha sin depender de `distance_center`.
3. **Ignora la curvatura**: Ya tienes polinomios cuadráticos con información de curvatura, pero la tiras. Si adelante hay curva, el robot no lo sabe hasta estar encima.
4. **Resolución entera**: Publicas `UInt8` (0-255). En la zona útil (~70-110°) son 40 valores. Muy grueso para control fino.

**Solución: publicar 3 señales float en un solo mensaje**

```cpp
#include <geometry_msgs/msg/vector3.hpp>

rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr control_ref_pub_;

// En constructor:
control_ref_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
    "/lane_detection/control_ref", 10);
```

### Señal 1: CTE (Cross-Track Error) — error lateral normalizado

Va de -1.0 (totalmente a la izquierda) a +1.0 (totalmente a la derecha). 0.0 = centrado perfecto.

```cpp
float cte = (float)(center_cam - center_lines) / (float)(lane_width_px_ / 2);
cte = std::clamp(cte, -1.0f, 1.0f);
```

### Señal 2: Heading error — error angular con signo

Diferencia de ángulo entre la dirección del robot y la tangente del carril.

```cpp
float row_near = PROC_HEIGHT - 1;
float row_far  = PROC_HEIGHT * 0.3f;

float center_near = 0.5f * (draw_left[0] + draw_left[1]*row_near + draw_left[2]*row_near*row_near
                           + draw_right[0] + draw_right[1]*row_near + draw_right[2]*row_near*row_near);
float center_far  = 0.5f * (draw_left[0] + draw_left[1]*row_far + draw_left[2]*row_far*row_far
                           + draw_right[0] + draw_right[1]*row_far + draw_right[2]*row_far*row_far);

float lane_dx = center_far - center_near;
float lane_dy = row_far - row_near;
float lane_angle = atan2f(lane_dx, -lane_dy);

float heading_error = std::clamp(lane_angle / 0.7854f, -1.0f, 1.0f);
```

### Señal 3: Curvatura anticipada (lookahead)

Permite al controlador pre-girar antes de entrar a la curva.

```cpp
float avg_p2 = 0.5f * (draw_left[2] + draw_right[2]);
float curvature = std::clamp(avg_p2 / 0.003f, -1.0f, 1.0f);
```

### Publicación unificada

```cpp
geometry_msgs::msg::Vector3 ref;
ref.x = cte;             // Error lateral [-1, 1]
ref.y = heading_error;   // Error angular [-1, 1]
ref.z = curvature;       // Curvatura anticipada [-1, 1]
control_ref_pub_->publish(ref);
```

### Controlador futuro: 3 términos paralelos + feedforward

Con estas 3 señales, el controlador puede usar:

```cpp
void control_callback(const geometry_msgs::msg::Vector3::SharedPtr ref) {
    float cte = ref->x;
    float heading = ref->y;
    float curvature = ref->z;

    // P sobre CTE (mantener centrado)
    float p_cte = Kp_cte * cte;

    // D sobre CTE (amortiguar oscilaciones laterales, opcional)
    float d_cte = Kd_cte * (cte - prev_cte_) / dt;
    prev_cte_ = cte;

    // P sobre heading (alinear con el carril)
    float p_heading = Kp_heading * heading;

    // Feedforward de curvatura (anticipar la curva)
    float ff_curvature = Kff * curvature;

    // Comando combinado
    float steering = p_cte + d_cte + p_heading + ff_curvature;
    steering = std::clamp(steering, -1.0f, 1.0f);

    // Velocidad: reducir en curvas
    float speed = max_speed_ * (1.0f - 0.5f * std::abs(curvature));
}
```

**Variables a calibrar**: 4 (o 5 con Kd_cte opcional). Cada una tiene un efecto visible y aislado:
- **Kp_cte** — no se centra → subirlo
- **Kp_heading** — zigzaguea → subirlo
- **Kff** — entra tarde a curvas → subirlo
- **max_speed** — velocidad máxima en recta

Se calibran secuencialmente, no se afectan entre sí.

**Ventajas**:

| Aspecto | Método actual | CTE + heading + curvatura |
|---------|--------------|---------------------------|
| **Señales** | 2 (ángulo int, distancia int) | 3 floats normalizados |
| **Signo** | Se pierde (abs) | Conservado |
| **Anticipación** | Ninguna | Curvatura lookahead |
| **Resolución** | ~40 valores (UInt8 70-110) | Float32 continuo |
| **Controlador** | PD reactivo | P+P+feedforward predictivo |
| **Velocidad adaptativa** | No | Sí (basada en curvatura) |

### Mantener compatibilidad durante transición

```cpp
// Seguir publicando los mensajes viejos
center_message.data = distance_center;
center_pub->publish(center_message);
angle_line_message.data = angle;
angle_line_pub->publish(angle_line_message);

// Y TAMBIÉN las señales nuevas (para ir probando con rosbag)
geometry_msgs::msg::Vector3 ref;
ref.x = cte;
ref.y = heading_error;
ref.z = curvature;
control_ref_pub_->publish(ref);
```

---

## Resumen y orden de implementación


**Flujo operativo final**: Colocar robot en pista → lanzar nodo → 2 seg de calibración automática (robot quieto, mide ancho y color) → conduce con iluminación adaptándose continuamente → si quitan una línea, infiere la otra con ancho adaptativo → si quitan ambas, navegación inercial (sigue derecho con decaimiento suave) → cuando reaparecen, transición gradual sin saltos → el controlador recibe CTE, heading y curvatura para control predictivo con feedforward, y ajusta velocidad según curvatura anticipada.