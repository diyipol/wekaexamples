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



## Entorno

Este tutorial está escrito usando el siguiente entorno:

- Hardware: MacBook Pro 15’ (2,5 GHz Intel Core i7, 16GB DDR3)
- Sistema operativo: macOS Mojave 10.14.1
- Versiones del software:
  - Weka: 3.9
  - JDK: 1.8



## Ejemplo de clasificación

Vamos a empezar con la técnica de _learning machine_ más utilizada, la clasificación. 

Todo el código de este tutorial puede ser descargado desde https://github.com/diyipol/wekaexamples.git. Para este ejemplo vamos a utilizar el fichero "weather.numeric.arff" que viene en los ficheros de ejemplo  dentro de la carpeta "data" de la distribución de _Weka_. En este ejemplo tenemos como variable objetivo si se puede jugar o no al golf dependiendo de las condiciones meteorológicas.

Creamos un proyecto maven y le añadimos la dependencia de weka. 

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
String[] opts = new String[]{"-R", "2"};
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

El proceso de decisión comienza en el nodo raíz, en este caso con el atributo _outlook_. Vemos hay dos instancias que indican que se puede jugar al golf si está soleado y la humedad es menor al 75%. Dos instancias indican que no se puede jugar al golf si está soleado pero la humedad es mayor al 75% y así podemos seguir recorriendo todo el árbol.

Ahora vamos a clasificar una nueva instancia. Para ello vamos a usar la clase _DenseInstance_. Iremos estableciendo el valor de cada atributo según su índice. Finamente le pediremos al árbol J48 que nos lo clasifique. 

```java
Instance instance = new DenseInstance(4);
instance.setDataset(instances);
instance.setValue(0, "sunny");
instance.setValue(1, 65);
instance.setValue(2, 80);
instance.setValue(3, "TRUE");

double result = tree.classifyInstance(instance);

System.out.println("Resultado de clasificar la nueva instancia:" + result);
```

Esta clase sólo trabaja con números flotantes, por lo que si el atributo es no numeral lo almacena/muestra según el índice con que se haya definido en las instancias. En este caso como la humedad es mayor al 75% (índice 2) vemos que el resultado es 1.0. Como habíamos definido el atributo _play_ de la siguiente manera:

```
@attribute play {yes, no}
```

Ese 1.0 sabemos que significa NO.

Si seteamos el valor con índice 2 a 65 veremos como la salida cambia a 0.0 (yes).

```java
instance.setValue(2, 65);
```

```shell
Resultado de clasificar la nueva instancia:0.0
```



### Métricas de error de evaluación y predicción

Ahora vamos a ver qué confianza podemos tener en el modelo que acabamos de construir. Para estimar su rendimiento podemos usar la técnica de validación cruzada (cross-validation).

Con la validación cruzada vamos dividiendo las instancias de muestra en K subconjuntos. Usaremos una parte del conjunto para entrenar y otra para test.  Esto se repetirá en K iteraciones para finalmente hallar la media aritmética. Por ejemplo en nuestro caso, que tenemos 14 instancias podemos:

* 1ª iteración: Datos de tests -> 1, 2 y 3. Datos de entrenamiento -> 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 y 14
* 2ª iteración: Datos de tests -> 4, 5 y 6. Datos de entrenamiento -> 1, 2, 3, 7, 8, 9, 10, 11, 12, 13 y 14
* 3ª iteración: Datos de tests -> 7, 8 y 9. Datos de entrenamiento -> 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13 y 14
* Y así hasta la iteración K.

De esta manera, utilizamos todos los datos para aprender y testear, mientras evitamos usar los mismos datos para entrenar y probar un modelo.

Para nuestro ejemplo vamos a dividir en 5 subconjuntos.

```java
Classifier classifier = new J48();
Evaluation evaluation = new Evaluation(instances);
int numFolds = 5;
Random random = new Random(1);
evaluation.crossValidateModel(classifier, instances, numFolds, random, new Object[] {});
System.out.println(evaluation.toSummaryString());
```

Hay que tener en cuenta que la salida que obtenemos no diferencia entre clasificación y regresión:

```shell
Correctly Classified Instances           9               64.2857 %
Incorrectly Classified Instances         5               35.7143 %
Kappa statistic                          0.186 
Mean absolute error                      0.3214
Root mean squared error                  0.5428
Relative absolute error                 68.8235 %
Root relative squared error            112.8506 %
Total Number of Instances               14    
```

Por lo que en este caso sólo nos interesa el número de instancias correctamente e incorrectamente clasificadas, que en este caso podemos observar que  se han acertado 9 de las 14 instancias.

Lo siguiente que vamos a inspeccionar es dónde se ha realizado una clasificación errónea con la matríz de confusión. La podemos obtener en un array llamando a:

```java
double[][] confusionMatrix = evaluation.confusionMatrix();
```

O imprimirla directamente con:

```java
System.out.println(evaluation.toMatrixString());
```

El resultado que obtenemos al imprimirla es:

```shell
=== Confusion Matrix ===

 a b   <-- classified as
 7 2 | a = yes
 3 2 | b = no
```

En este ejemplo a tener una salida booleana no nos aporta ninguna información esta matriz, pero en un ejemplo más complejo, por ejemplo para adivinar animales a partir de sus rasgos (https://github.com/fracpete/collective-classification-weka-package/blob/master/src/site/resources/datasets/nominal/zoo.arff) podemos observar como a un reptil en una ocasión lo clasificó como insecto y en otra como un pez.

```shell
    === Confusion Matrix ===
    
      a  b  c  d  e  f  g   <-- classified as
     41  0  0  0  0  0  0 |  a = mammal
      0 20  0  0  0  0  0 |  b = bird
      0  0  3  1  0  1  0 |  c = reptile
      0  0  0 13  0  0  0 |  d = fish
      0  0  1  0  3  0  0 |  e = amphibian
      0  0  0  0  0  5  3 |  f = insect
      0  0  0  0  0  2  8 |  g = invertebrate
```



### Elección de un algoritmo

Naive Bayes es uno de los algoritmos más simples, eficientes y efectivos en _machine learning_. Cuando las características son independientes, algo raro en el mundo real, teóricamente es óptimo. Aún así, incluso con atributos dependientes, es muy competitivo. Su principal desventaja  es la incapacidad que tiene de aprender cómo las características interactúan entre sí. Por ejemplo te puede gustar el café y te pueden gustar las tartas, pero odías las tartas con sabor a café. 

Por otra parte, como hemos podido comprobar. la principal ventaja del árbol de decisión es que es un modelo muy fácil de entender y explicar. Además puede manejar atributos tanto categorizados como numéricos y requiere poca preparación de los datos.

Para el mismo ejemplo vamos a ver el porcentaje de aciertos de Naive Bayes.

```java
Classifier naiveBayesClassifier = new NaiveBayes();
Evaluation naiveBayesEvaluation = new Evaluation(instances);
naiveBayesEvaluation.crossValidateModel(naiveBayesClassifier, instances, numFolds, random, new Object[] {});
System.out.println(naiveBayesEvaluation.toSummaryString());
```

Vemos que en este caso, el porcentaje de errores es demasiado elevado, por lo que el árbol de decisión es mejor opción.

```shell
Correctly Classified Instances           4               28.5714 %
Incorrectly Classified Instances        10               71.4286 %
Kappa statistic                         -0.4286
Mean absolute error                      0.6016
Root mean squared error                  0.6325
Relative absolute error                128.8169 %
Root relative squared error            131.4982 %
Total Number of Instances               14     
```



## Ejemplo de regresión

Para el ejemplo de regresión vamos a usar el fichero "housing.arff" que podemos descargar del repositorio de con los ejemplos de este tutorial (https://github.com/diyipol/wekaexamples/blob/master/src/main/resources/housing.arff).

En el fichero de ejemplo nos encontramos con 506 registros que nos describen precios de la vivienda en los suburbios de Houston. De los catorce atributos que hay, trece son contínuos y uno binario. Nos encontramos atributos del tipo: tasa de criminalidad per cápita por ciudad, proporción de tierra residencial zonificada, concentración de óxidos nítricos, número medio de habitaciones por vivienda, etc. La variable objetivo es MEDV (Valor medio de las casas ocupadas por sus propietarios en $1000).

Al igual que en el ejemplo anterior cargamos el fichero mediante el _DataSource_ para luego convertirlo a instancias y estableciendo el atributo de clase.

```java
File file = fileUtils.getFile("housing.arff");
ConverterUtils.DataSource source = new ConverterUtils.DataSource(file.getAbsolutePath());
Instances instances = source.getDataSet();
instances.setClassIndex(instances.numAttributes() - 1);
```



### Regresión lineal 

Vamos a comenzar con la regresión más sencilla, la lineal, que supone una dependencia lineas entre las características y la variable objetivo. En muchos casos este tipo de regresión no es capaz de modelar relaciones complejas.

Al igual que con la clasificación vamos a tener una clase que nos implemente el algoritmo de regresión, en este caso la clase _LinearRegression_.

```java
import weka.classifiers.functions.LinearRegression;
...
    
LinearRegression linearRegression = new LinearRegression();
linearRegression.buildClassifier(instances);

System.out.println(linearRegression);
```

 Como de costumbre, el _toString_ nos imprime nuestro modelo:

```shell
Linear Regression Model

class =

     -0.1084 * CRIM +
      0.0458 * ZN +
      2.7187 * CHAS=1 +
    -17.376  * NOX +
      3.8016 * RM +
     -1.4927 * DIS +
      0.2996 * RAD +
     -0.0118 * TAX +
     -0.9465 * PTRATIO +
      0.0093 * B +
     -0.5226 * LSTAT +
     36.3411

```

Vemos que nos ha construido una función lineal que combina las variables de entrada para calcular el valor medio de la casa. Vamos a ver la validación cruzada con 10 subconjuntos.

```shell
Correlation coefficient                  0.8451
Mean absolute error                      3.3933
Root mean squared error                  4.9145
Relative absolute error                 50.8946 %
Root relative squared error             53.3085 %
Total Number of Instances              506    
```



### Árboles de regresión

Otra aproximación es construir un conjunto de modelos de regresión, cada uno con su propia parte de los datos.

![](/Users/pablojose.betancor/Desarrollo/workspaces/intellij/wekaexamples/images/árboles de regresión.png)

En Weka los árboles de regresión están implementados en la clase M5:

```java
M5P regressionTree = new M5P();
regressionTree.setOptions(new String[]{""});
regressionTree.buildClassifier(instances);
System.out.println(regressionTree);
```

El modelo inducido es un árbol con ecuaciones en los nodos hoja:

```shell
M5 pruned model tree:
(using smoothed linear models)

LSTAT <= 9.725 : 
|   RM <= 6.941 : 
|   |   DIS <= 3.325 : 
|   |   |   RAD <= 7.5 : LM1 (23/38.466%)
|   |   |   RAD >  7.5 : 
|   |   |   |   CRIM <= 4.727 : LM2 (3/22.662%)
|   |   |   |   CRIM >  4.727 : LM3 (4/0%)
|   |   DIS >  3.325 : 
|   |   |   RM <= 6.545 : LM4 (72/15.074%)
|   |   |   RM >  6.545 : 
|   |   |   |   LSTAT <= 4.915 : 
|   |   |   |   |   PTRATIO <= 17.75 : LM5 (11/8.841%)
|   |   |   |   |   PTRATIO >  17.75 : LM6 (4/10.82%)
|   |   |   |   LSTAT >  4.915 : 
|   |   |   |   |   RM <= 6.611 : LM7 (7/9.288%)
|   |   |   |   |   RM >  6.611 : LM8 (18/14.58%)
|   RM >  6.941 : 
|   |   RM <= 7.437 : 
|   |   |   AGE <= 76.95 : LM9 (29/22.763%)
|   |   |   AGE >  76.95 : 
|   |   |   |   B <= 394.7 : LM10 (8/40.649%)
|   |   |   |   B >  394.7 : LM11 (3/6.55%)
|   |   RM >  7.437 : LM12 (30/35.25%)
LSTAT >  9.725 : 
|   LSTAT <= 15 : LM13 (132/28.25%)
|   LSTAT >  15 : 
|   |   CRIM <= 5.769 : 
|   |   |   CRIM <= 0.654 : 
|   |   |   |   DIS <= 1.906 : LM14 (10/16.417%)
|   |   |   |   DIS >  1.906 : LM15 (36/26.564%)
|   |   |   CRIM >  0.654 : LM16 (37/22.767%)
|   |   CRIM >  5.769 : 
|   |   |   LSTAT <= 19.73 : LM17 (29/17.632%)
|   |   |   LSTAT >  19.73 : 
|   |   |   |   NOX <= 0.675 : LM18 (16/35.377%)
|   |   |   |   NOX >  0.675 : LM19 (34/15.449%)

LM num: 1
class = 
	4.7924 * CRIM 
	+ 0.003 * ZN 
	- 0.4795 * INDUS 
	- 8.2717 * NOX 
	+ 3.1363 * RM 
	- 0.4092 * DIS 
	+ 0.1121 * RAD 
	- 0.0075 * TAX 
	- 0.2873 * PTRATIO 
	+ 0.0006 * B 
	- 0.742 * LSTAT 
	+ 25.4161
	
...

LM num: 19
class = 
	-0.0829 * CRIM 
	+ 0.0022 * ZN 
	- 15.0168 * NOX 
	- 0.5959 * RM 
	+ 0.0293 * AGE 
	+ 1.6442 * DIS 
	+ 0.0227 * RAD 
	- 0.0014 * TAX 
	- 0.1031 * PTRATIO 
	+ 0.0004 * B 
	- 0.2522 * LSTAT 
	+ 28.8208

Number of Rules : 19
```

En este caso el árbol tiene 19 hojas donde cada una se corresponde a una ecuación lineal. Vamos a comprobar ahora si mejoramos la validación.

```java
Evaluation regressionTreeEvaluation = new Evaluation(instances);
regressionTreeEvaluation.crossValidateModel(regressionTree, instances, numFolds, random, new Object[]{});
System.out.println(regressionTreeEvaluation.toSummaryString());
```

Vemos que hemos mejorado un poco:

```shell
Correlation coefficient                  0.9155
Mean absolute error                      2.4485
Root mean squared error                  3.6975
Relative absolute error                 36.6528 %
Root relative squared error             40.0259 %
Total Number of Instances              506     
```



## Referencias

* Machine Learning in Java - Second Edition by AshishSingh Bhatia; Bostjan Kaluza. Published by Packt Publishing, 2018.
* https://www.cs.waikato.ac.nz/ml/weka/