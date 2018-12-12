## Introducción

El _"machine learning"_ son los algoritmos y técnicas que se utilizan en la fase de análisis y modelado del proceso de _"data science"_. "_Data science"_ abarca todo el proceso de obtención de conocimiento, limpieza, análisis, visualización y despliegue de de datos.

Dentro del _machine learning_ existen tres tipos de aprendizaje:

* Aprendizaje supervisado, donde el algoritmo se entrena con datos históricos. Por ejemplo la detección de fraudes de tarjetas de crédito, donde a partir de transacciones de tarjetas marcadas como normales o sospechosas, el algoritmo produce una predicción sobre una nueva entrada.
* Aprendizaje no supervisado donde no se disponen de datos para el entrenamiento. Por ejemplo un sistema de recomendaciones que descubre patrones ocultos en los datos sin asumir etiquetas de resultados determinadas.
* Aprendizaje reforzado. Por ejemplo un software que conduce un vehículo, donde el programa interactúa con un entorno dinámico para lograr un objetivo específico.

Dentro de estos, nos vamos a centrar en el aprendizaje supervisado, que se suele usar para problemas de clasificación y de regresión.



### Clasificación

La clasificación se puede aplicar cuando tratamos con una clase discreta, donde el objetivo es predecir uno de los valores mutuamente exclusivos en la variable objetivo. Por ejemplo saber si va a llover o no. 

Los algoritmos de clasificación más populares son:

* Árboles de decisión
* Clasificación de Naive Bayes
* Support Vector Machines (SVMs)
* Redes neuronales
* Ensembles



### Regresión

Por otro lado, a diferencia de la clasificación, la regresión se aplica a una varible objetivo continua. Por ejemplo, en vez de saber si va a llover o no, para pronosticar la temperatura que hará.

Podemos destacar:

* Refresión lineal
* Refresión logística
* Regresión por mínimos cuadrados



### Weka

_Waikato Environment for Knowledge Analysis_ (WEKA) es una librería Java de _machine learning_ desarrollada en la Universidad de Waikato, Nueva Zelanda. Weka puede resolver una amplia variedad de tareas de _machine learning_ como la clasificación, la regresión y el clustering. Es software de código abierto emitido bajo la GNU General Public License.

Weka soporta varios formatos de ficheros, entre ellos tiene el suyo propio, **ARFF**. El formato tiene dos partes. La primera contiene encabezado, que especifica todos los atributos y sus tipos, por ejemplo, nominal, numérico, fecha y cadena de texto. La segunda parte contiene los datos, donde cada línea corresponde a una instancia. El último atributo en el encabezado se considera implícitamente la variable objetivo y los datos faltantes se marcan con un signo de interrogación.

Ejemplo:

```
@relation weather

@attribute outlook {sunny, overcast, rainy}
@attribute temperature numeric
@attribute humidity numeric
@attribute windy {TRUE, FALSE}
@attribute play {yes, no}

@data
sunny,85,85,FALSE,no
sunny,80,90,TRUE,no
overcast,83,86,FALSE,yes
rainy,70,96,FALSE,yes
rainy,68,80,FALSE,yes
rainy,65,70,TRUE,no
```



## Datos de entrada

Para el tutorial vamos a utilizar el fichero "weather.numeric.arff" que viene dentro de los ficheros de ejemplo dentro de la carpeta "data" de la distribución de _Weka_. En este ejemplo tenemos como variable objetivo si se puede jugar o no al golf dependiendo de las condiciones meteorológicas.



## Ejemplo de clasificación

Vamos a empezar con la técnica de _learning machine_ más utilizada, la clasificación. Creamos un proyecto maven y le añadimos la dependencia de weka. 

```xml
<dependency>
    <groupId>nz.ac.waikato.cms.weka</groupId>
    <artifactId>weka-dev</artifactId>
    <version>3.9.3</version>
</dependency>
```

 

En la carpeta de _resources_ añadimos el fichero _weather.numeric.arff_. Para cargar el fichero usaremos la clase _Datasource_ que soporta diferentes formatos de ficheros para transformarlos a la clase _Instances_. A las instancias les señalamos cual es la variable de salida (_class index_), en este caso va a ser el último atributo, _play_.

```java
File file = getFile("weather.arff");

ConverterUtils.DataSource source = new ConverterUtils.DataSource(file.getAbsolutePath());
Instances instances = source.getDataSet();
instances.setClassIndex(instances.numAttributes() - 1);

System.out.println(instances.numInstances() + " instancias cargadas.");
System.out.println(instances.toString());
```



Si necesitáramos eliminar algún atributo de las instancias podemos usar el filtro _Remove_. Por ejemplo para eliminar el atributo de temperatura que tiene el índice 2, haríamos:

```java
Remove remove = new Remove();
String[] opts = new String[]{ "-R", "2"};
remove.setOptions(opts);
remove.setInputFormat(instances);

instances = Filter.useFilter(instances, remove);
```

En nuestro ejemplo no vamos a eliminar ningún atributo.



### Árbol de decisión

En Weka el árbol de decisión está implementado en la clase J48, que es una reimplementación del algoritmo C4.5 de Quinlan (https://es.wikipedia.org/wiki/C4.5).

Al algoritmo le podemos pasar parámetros adicionales como por ejemplo la poda del árbol para controlar la complejidad del modelo. En este caso le vamos a indicar que no queremos podado. Finalmente para inicializar el proceso de aprendizaje llamamos a _buildClassifier_. El objeto construido se encuentra ahora en la variable _tree_ y podemos visualizarlo con _toString_.

```java
J48 tree = new J48();
String[] options = new String[1];
options[0] = TREE_UNPRUNED_OPT;

tree.setOptions(options);

tree.buildClassifier(instances);

System.out.println(tree);
```

Tenemos como salida:

```shell
J48 unpruned tree
------------------

outlook = sunny
|   humidity <= 75: yes (2.0)
|   humidity > 75: no (3.0)
outlook = overcast: yes (4.0)
outlook = rainy
|   windy = TRUE: no (2.0)
|   windy = FALSE: yes (3.0)

Number of Leaves  : 	5

Size of the tree : 	8
```



## Entorno

Este tutorial está escrito usando el siguiente entorno:

- Hardware: MacBook Pro 15’ (2,5 GHz Intel Core i7, 16GB DDR3)
- Sistema operativo: macOS Mojave 10.14.1
- Versiones del software:



## Referencias

* Machine Learning in Java - Second Edition by AshishSingh Bhatia; Bostjan Kaluza. Published by Packt Publishing, 2018.
* https://www.cs.waikato.ac.nz/ml/weka/