# Documentación Técnica Avanzada: Nodo de Detección de Objetos (Lidar PCL)

Este documento presenta un análisis exhaustivo, teórico y práctico del sistema de percepción implementado para el robot **Yahboom ROSMASTER A1**. Se detalla la fundamentación matemática de los algoritmos de Aprendizaje Automático No Supervisado utilizados para procesar las nubes de puntos del sensor **Slamtec C1**.

## 1. Contextualización del Problema

### 1.1 El Desafío de la Percepción en Robótica Móvil
El robot opera en entornos no estructurados donde la posición y geometría de los obstáculos son desconocidas *a priori*. El sensor LiDAR Slamtec C1 proporciona una representación discreta del entorno mediante un conjunto de puntos $P = \{p_1, p_2, ..., p_n\}$ en coordenadas polares $(r, \theta)$.

El problema fundamental es la **segmentación semántica no supervisada**: transformar este conjunto de puntos crudos $P$ en un conjunto de subconjuntos disjuntos $C = \{C_1, C_2, ..., C_k\}$, donde cada $C_i$ representa un objeto físico distinto (e.g., una pared, una persona, una caja).

### 1.2 Justificación del Enfoque de ML
Dado que no existen etiquetas de entrenamiento (ground truth) en tiempo real, se descartan los métodos supervisados. Se requiere un enfoque de **Clustering Basado en Densidad**, ya que los objetos físicos se manifiestan como regiones de alta densidad de puntos separadas por regiones de baja densidad (espacio vacío).

---

## 2. Fundamentación Teórica y Algorítmica

### 2.1 Preprocesamiento y Filtrado
Antes de la segmentación, los datos deben ser limpiados para reducir la complejidad computacional y eliminar errores de medición.

#### A. Filtrado Voxel Grid (Downsampling)
Para reducir la cantidad de puntos $N$ sin perder la estructura geométrica, se aplica un filtro de rejilla de vóxeles.
*   **Concepto:** El espacio 3D se divide en cubos (vóxeles) de tamaño **V_size × V_size × V_size**.
*   **Operación:** Todos los puntos dentro de un vóxel se reemplazan por su centroide **p̄**.
    
    > **p̄ = (1/m) · Σ pᵢ**
    
    Donde **m** es el número de puntos en el vóxel.
*   **Impacto en Código:** Controlado por `voxel_leaf_size_` (0.02m). Esto reduce drásticamente **N**, permitiendo que el algoritmo de clustering posterior corra en tiempo real.

#### B. Statistical Outlier Removal (Eliminación de Ruido)
El sensor LiDAR a menudo genera "puntos fantasma" debido a reflejos especulares o bordes de objetos.
*   **Algoritmo:** Para cada punto **pᵢ**, se calcula la distancia media **d̄ᵢ** a sus **k** vecinos más cercanos.
*   **Criterio de Eliminación:** Un punto se descarta si su distancia media es mayor que un umbral definido por la media global **μₖ** y la desviación estándar **σₖ** de todas las distancias:
    
    > **Descartar si: d̄ᵢ > μₖ + α · σₖ**

*   **Parámetros en Código:**
    *   **k** = `outlier_mean_k` (10 vecinos).
    *   **α** = `outlier_stddev` (0.5 desviaciones estándar).

### 2.2 Núcleo de ML: Euclidean Cluster Extraction (DBSCAN)
El algoritmo implementado es una variante eficiente de **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**. A diferencia de K-Means, este algoritmo agrupa puntos basándose en la conectividad espacial directa.

#### Formulación Matemática
Sea **d(pᵢ, pⱼ)** la distancia euclidiana entre dos puntos:

> **d(pᵢ, pⱼ) = √((xᵢ-xⱼ)² + (yᵢ-yⱼ)² + (zᵢ-zⱼ)²)**

Un cluster **C** se define como un conjunto de puntos tal que para todo **p ∈ C**, existe otro punto **q ∈ C** que cumple:

> **d(p, q) < ε**

Donde **ε** es la tolerancia de distancia (radio de búsqueda).

#### Algoritmo Implementado (PCL)
1.  Se selecciona un punto semilla $p$ no procesado.
2.  Se buscan todos los vecinos **N_ε(p)** tal que **d(p, n) < ε**.
3.  Si el número de vecinos **|N_ε(p)| ≥ MinPts**, se crea un nuevo cluster.
4.  Se repite el proceso recursivamente para cada vecino, expandiendo el cluster hasta que no se encuentren más puntos conectados por la distancia **ε**.

#### Optimización con Kd-Tree
La búsqueda ingenua de vecinos tiene una complejidad de $O(N^2)$. Para hacer esto viable en el robot, se utiliza una estructura de datos **Kd-Tree (k-dimensional tree)**.
*   Esto reduce la complejidad de búsqueda del vecino más cercano a $O(\log N)$.
*   **Complejidad total del algoritmo:** $O(N \log N)$.

---

## 3. Implementación: Mapeo de Teoría a Código

La siguiente tabla relaciona los conceptos teóricos explicados con las variables específicas en `object_detection_pcl.cpp`:

| Concepto Teórico | Variable en Código | Valor Configurado | Explicación |
| :--- | :--- | :--- | :--- |
| **Radio de Búsqueda (ε)** | `cluster_tolerance_` | `0.15` (metros) | Distancia máxima entre puntos para considerarlos parte del mismo objeto. Si dos puntos están a más de 15cm, se consideran objetos separados. |
| **Densidad Mínima (MinPts)** | `min_cluster_size_` | `3` (puntos) | Mínimo de puntos para formar un cluster válido. Ayuda a filtrar ruido residual que no fue eliminado por el filtro estadístico. |
| **Tamaño Máximo** | `max_cluster_size_` | `10000` (puntos) | Evita que el algoritmo agrupe todo el entorno (e.g., paredes largas) como un único objeto inmanejable. |
| **Región de Interés (ROI)** | `range_min_`, `range_max_` | `0.1`m - `16.0`m | Filtra puntos demasiado cercanos (ruido del propio chasis) o demasiado lejanos (fuera de relevancia para navegación local). |

### 3.1 Ajuste de Hiperparámetros (Tuning)
Los valores configurados en el código no son arbitrarios; son el resultado de un proceso de **sintonización empírica (manual tuning)** realizado mediante pruebas iterativas con el robot real.

*   **Cluster Tolerance (0.15m):** Se ajustó para equilibrar la segmentación. Un valor menor fragmentaba objetos grandes (paredes) en múltiples clusters, mientras que un valor mayor fusionaba objetos distintos (dos sillas cercanas).
*   **Min Cluster Size (3 pts):** Se determinó experimentalmente para filtrar el ruido de "salt & pepper" típico del sensor Slamtec C1 sin perder objetos pequeños pero legítimos.
*   **Voxel Leaf Size (0.02m):** Optimizado para mantener la forma geométrica de los objetos reduciendo la carga de CPU al mínimo necesario para operar a 10+ FPS.

Estos parámetros representan la configuración óptima encontrada para el entorno de operación del Yahboomcar.

---

## 4. Análisis Comparativo: ¿Por qué no K-Means?

Es crucial justificar por qué se eligió un método basado en densidad sobre el popular K-Means.

### 1. Conocimiento a Priori del Entorno
*   **K-Means:** Requiere definir $K$ (número de clusters) antes de ejecutar.
    *   *Problema:* En navegación, el robot no sabe si tiene enfrente 1, 5 o 0 obstáculos. Un $K$ incorrecto forzaría al algoritmo a dividir una pared en dos objetos o fusionar dos sillas en una.
*   **DBSCAN/Euclidean:** No requiere $K$. Descubre el número de objetos automáticamente basado en la estructura de los datos.

### 2. Geometría de los Objetos
*   **K-Means:** Asume que los clusters son convexos (esféricos) e isotrópicos. Utiliza la distancia al centroide.
    *   *Problema:* Una pared larga y delgada sería mal clasificada por K-Means, ya que los puntos en los extremos están lejos del centroide.
*   **DBSCAN/Euclidean:** Basado en conectividad local. Puede seguir formas arbitrarias (serpenteantes, líneas, "U" shapes) perfectamente, lo cual es ideal para paredes y esquinas.

### 3. Robustez al Ruido
*   **K-Means:** Es sensible a outliers. Un solo punto de ruido lejos de los objetos desplazará significativamente el centroide del cluster más cercano.
*   **DBSCAN/Euclidean:** Tiene una noción explícita de "ruido". Los puntos que no cumplen el criterio de densidad no se asignan a ningún cluster, resultando en una detección mucho más limpia.

## 5. Flujo de Datos del Nodo

```mermaid
graph TD
    subgraph "Hardware & Drivers"
        LIDAR["Slamtec C1"] -->|Scan Raw| ROS2["ROS2 Driver"]
    end
    
    subgraph "Nodo: Object Detection PCL"
        ROS2 -->|/scan| SUB["Suscripción LaserScan"]
        SUB --> CONV["Conversión Polar a Cartesiana"]
        
        subgraph "Preprocesamiento"
            CONV --> ROI["Filtro de Rango y Ángulo"]
            ROI --> VOXEL["Voxel Grid Filter (Downsampling)"]
            VOXEL --> SOR["Statistical Outlier Removal"]
        end
        
        subgraph "Core ML"
            SOR --> KDTREE["Construcción Kd-Tree"]
            KDTREE --> EUCL["Euclidean Cluster Extraction"]
        end
        
        subgraph "Post-Procesamiento"
            EUCL --> FEAT["Cálculo de Centroides y BBox"]
            FEAT --> VIS["Visualización OpenCV"]
            FEAT --> PUB["Publicación PointCloud2"]
        end
    end
    
    PUB -->|/detected_objects_cloud| NAV["Stack de Navegación"]
```

### 5.1 Detalle del Algoritmo DBSCAN (Euclidean Clustering)

El siguiente diagrama profundiza en la lógica interna del bloque "Core ML", ilustrando cómo el algoritmo procesa cada punto para formar clusters.

```mermaid
graph TD
    %% Subgraph 1: Adquisición
    subgraph "1. Adquisición de Datos"
        Input([Entrada: sensor_msgs/LaserScan])
        Param[Parámetros: ε=0.15m, MinPts=3, Voxel=0.02m]
        PolarToCart["Conversión Polar -> Cartesiana (x,y,z)"]
        Input --> PolarToCart
        PolarToCart --> CloudRaw["pcl::PointCloud<pcl::PointXYZ> (Raw)"]
    end

    %% Subgraph 2: Preprocesamiento Detallado
    subgraph "2. Limpieza y Preprocesamiento"
        CloudRaw --> VoxelGrid["Voxel Grid Filter"]
        VoxelGrid -->|Divide espacio en cubos 2cm³| ComputeCentroid["Calcular Centroide por Voxel: p̄ = Σpᵢ / m"]
        ComputeCentroid --> CloudVoxel["Nube Reducida (Downsampled)"]
        
        CloudVoxel --> SOR["Statistical Outlier Removal"]
        SOR --> CalcMeanDist["Calcular Distancia Media a k=10 vecinos (d̄)"]
        CalcMeanDist --> CheckThresh{"¿d̄ > μ + α·σ?"}
        CheckThresh -->|Sí| DiscardPoint["Eliminar Punto (Ruido)"]
        CheckThresh -->|No| KeepPoint["Mantener Punto"]
        KeepPoint --> CleanPoints["Nube Filtrada P'"]
    end

    %% Subgraph 3: Core ML (DBSCAN Low-Level)
    subgraph "3. Agrupamiento (Core ML - Euclidean Clustering)"
        CleanPoints --> BuildTree["Construir Kd-Tree (O(N log N))"]
        BuildTree --> InitLoop{¿Puntos sin procesar?}
        
        InitLoop -->|No| EndClustering[Fin del Agrupamiento]
        InitLoop -->|Sí| SelectSeed["Seleccionar Semilla p_i"]
        SelectSeed --> SearchNeighbors["Kd-Tree Radius Search (p_i, ε)"]
        SearchNeighbors --> CalcDist["Calc Distancia Euclidiana: d = √((x₁-x₂)²+...)"]
        CalcDist --> CheckEps{"¿d < ε?"}
        CheckEps -->|Sí| AddToN["Agregar a Vecinos N"]
        
        AddToN --> CheckMinPts{"¿|N| ≥ MinPts?"}
        CheckMinPts -->|No| MarkNoise["Marcar como RUIDO (temporal)"]
        MarkNoise --> InitLoop
        
        CheckMinPts -->|Sí| NewCluster["Crear Nuevo Cluster C_k"]
        NewCluster --> AddSeed["Agregar p_i a C_k"]
        AddSeed --> ExpandQueue["Cola de Procesamiento Q = N"]
        
        ExpandQueue --> QueueEmpty{¿Q vacía?}
        QueueEmpty -->|Sí| SaveCluster["Guardar C_k en lista de Clusters"]
        SaveCluster --> InitLoop
        
        QueueEmpty -->|No| PopQ["Extraer punto q de Q"]
        PopQ --> Visited{"¿q visitado?"}
        Visited -->|Sí| CheckInC{"¿q en algún C?"}
        CheckInC -->|No| AddToCk["Agregar q a C_k"]
        CheckInC -->|Sí| QueueEmpty
        
        Visited -->|No| MarkVisited["Marcar q VISITADO"]
        MarkVisited --> SearchQ["Kd-Tree Search (q, ε)"]
        SearchQ --> CheckDensityQ{"¿|N_q| ≥ MinPts?"}
        CheckDensityQ -->|Sí| AppendQ["Agregar N_q a Q"]
        CheckDensityQ -->|No| AddToCk
        AppendQ --> AddToCk
        AddToCk --> QueueEmpty
    end

    %% Subgraph 4: Evaluación
    subgraph "4. Evaluación del Modelo"
        SaveCluster --> EvalSize{"Validar Tamaño |C_k|"}
        EvalSize -->|"< 3 pts"| DiscardC["Descartar (Under-segmentation)"]
        EvalSize -->|"> 10000 pts"| SplitC["Descartar (Over-segmentation)"]
        EvalSize -->|"OK"| ValidC["Cluster Validado"]
    end

    %% Subgraph 5: Interpretación
    subgraph "5. Interpretación de Resultados"
        ValidC --> CalcCentroid["Cálculo Centroide: C = (Σx/n, Σy/n, Σz/n)"]
        CalcCentroid --> CalcMinMax["Buscar Min/Max (x,y,z) para BBox"]
        CalcMinMax --> CreateMsg["Generar sensor_msgs::PointCloud2"]
        CreateMsg --> Output(["Salida: /detected_objects_cloud"])
    end

    %% Subgraph 6: Validación Humana
    subgraph "6. Validación Empírica (Human-in-the-loop)"
        Output --> VisualCheck{"Inspección Visual (Rviz/OpenCV)"}
        VisualCheck -->|Falsos Positivos| Tune1[Ajustar MinPts/Epsilon]
        VisualCheck -->|Flickering| Tune2[Ajustar Filtros SOR]
        VisualCheck -->|FPS < 10| Tune3[Ajustar Voxel Size]
        Tune1 -.-> Param
        Tune2 -.-> Param
        Tune3 -.-> Param
        VisualCheck -->|OK| Final([Modelo Optimizado para Navegación])
    end
```

### 5.2 Metodología de Evaluación y Validación de Resultados
Dado que este es un sistema de aprendizaje no supervisado operando en un entorno real sin etiquetas (ground truth), la evaluación de la "optimalidad" del modelo se realizó mediante una metodología **Cualitativa y Empírica**:

1.  **Inspección Visual (Correlación Realidad-Modelo):**
    *   Se comparó la visualización generada en OpenCV con la disposición física real de los obstáculos.
    *   *Criterio de Éxito:* Cada objeto físico (caja, pared, persona) debe corresponder a un único Bounding Box en la visualización.

2.  **Prueba de Estabilidad Temporal:**
    *   Se observó la consistencia de los clusters sobre objetos estáticos.
    *   *Criterio de Éxito:* Los clusters no deben aparecer y desaparecer ("flickering") si el objeto y el robot están inmóviles.

3.  **Minimización de Falsos Positivos (Ruido):**
    *   Se ajustó `min_cluster_size` hasta eliminar detecciones en espacio vacío causadas por polvo o ruido del sensor.
    *   *Resultado:* Reducción de detecciones fantasma a < 1% en condiciones controladas.

4.  **Desempeño en Tiempo Real (Latencia):**
    *   Se monitoreó la tasa de cuadros (FPS) del nodo.
    *   *Validación:* El sistema mantiene >10 FPS constantes, lo cual es crítico para que el stack de navegación reaccione a tiempo para evitar colisiones.

## 6. Instrucciones de Uso

1.  **Iniciar Hardware:**
    ```bash
    ros2 launch sllidar_ros2 sllidar_c1_launch.py
    ```
2.  **Ejecutar Nodo:**
    ```bash
    ros2 run object_detection object_detection_node
    ```
3.  **Ajuste de Parámetros (Runtime):**
    El nodo permite reconfiguración dinámica. Si el robot detecta "falsos positivos" (ruido como objetos), aumente `min_cluster_size` o reduzca `cluster_tolerance`.
    ```bash
    ros2 param set /object_detection_pcl min_cluster_size 5
    ```

## 7. Resultados y Demostración

A continuación se muestra el funcionamiento del sistema en un entorno real.

### Visualización del Sistema
![Vista del Sistema de Detección](result_preview.png)

### Video de Demostración
Haga clic en la imagen o en el enlace para ver el video del funcionamiento en tiempo real:

[![Demo Video](https://img.youtube.com/vi/Fgq_LCIFgHU/0.jpg)](https://youtu.be/Fgq_LCIFgHU)

[Ver video en YouTube](https://youtu.be/Fgq_LCIFgHU)
