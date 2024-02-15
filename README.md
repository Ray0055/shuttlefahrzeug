本次的项目一共分为三部分,车道线识别, 转向控制和油门控制. 接下来依次进行说明.

# Lane Detection
## Region Detection
第一步是从摄像头接受的图像中裁剪出感兴趣的区域, 并进行初次的筛选. 

对于车道识别来说, 远处的天空, 对象车道的车道线对自己车道的识别只会产生干扰, 因此, 可以切出一个引擎盖之上, 左右两车道附近的一个梯形区域. 这一步用`cv::fillpoly`画出梯形区域, 使用 `cv::bitwise_and()`对图像进行裁切.

## Color Detection
第二步是从ROI区域进行颜色筛选. 

在道路图像中, 只需要关注白色或者黄色的车道线. 用HSV颜色空间即可实现筛选功能. 白线需要高亮度值，黄线需要特定的色调范围和饱和度。

值得注意的是在进行白线颜色筛选的时候, 如果地上有反光.积雪, 或者白色的物体, 都容易对识别的结果进行干扰. 因此使用hsv颜色空间只是针对道路情况简单, 如果道路情况复杂, 则还需要更高级的颜色筛选方法, 比如结合深度学习等等. 

## Edge Detection
在得到只有白色或者黄色的车道线识别图后, 可以对车道线的边缘进行识别. 

首先是进行图像的初步处理, 需要用 `cv::GaussianBlur()` 对图像进行降噪处理. 对边缘和拐角的识别方法有两种: `cv::Canny()` 和 `cv::Sobel()` 两者方法都可以识别出边缘和拐角, 但是在实际检测中, 前者的效果更好.

## Lane Detection
最后一步则是结合上一步识别出的边缘拟合出左右两车道. 

首先用 `cv::HoughLinesP()` 来识别出边缘上的线段. 通过调整函数中的参数, 会得到长度/宽度/距离不一样的线段群. 在这里可以尽量将参数中线段的长度阈值调整的大一点, 这样可以避免被道路中小的杂物或者光斑影响. 

得到合适的线段群之后, 需要进行左右车道的区分. 可以直接用slope的范围进行区分. 将正的斜率分为右车道, 负的斜率分为左车道. 值得注意的是, 如果仅仅通过正负区分车道是不够的, 如果识别到停止线或者斑马线等斜率特别小的线段, 也会被分入左车道或者右车道, 对车道的识别进行干扰. 

这里有个小技巧: 将识别出的raw lines的slope全部导入到csv文件中, 手动驾驶汽车跑完全图, 然后分析csv中slope的范围, 在我自己的情形中, 右车道的斜率范围为[0.3,0.75], 左车道的斜率范围为[-0.75, -0.2].

在得到左右车道raw_lines之后, 需要将这些线段拟合成一个. 在这里用到了 `cv::fitLine()`, 这个函数用最小二乘法的原理从点集中拟合出一根直线. 这根直线就是所识别出的左右两车道. 

为了使结果更加的鲁棒, 这里用所识别的直线的历史数据的平均值来代替当前所识别的直线. 效果很好, 如果不采用历史平均值, 由于每次识别的线段群不一样, 所拟合的车道线也会有细微的差距. 这个差距在实际应用的体现就是所识别的车道线一直在抖动. 

最后将所识别的车道线绘制到原图上即可.

# Steering Control
转向控制中, 采用PID控制器得到合适的转向值. 对于如何定义比例误差P, 有非常多的参考值可供选择: 
1. 通过识别的左右两车道, 可以计算出一条车道中心和车辆中心的连线. 这条线段表示了车辆的方向. 在正常行驶过程中, 该连线应该一直保持竖直. 因此, 可以用该连线和垂直方向的夹角作为比例误差P.
2. 通过识别的左右两车道, 可以计算出车辆中心与道路中心的误差. 在正常行驶过程中. 车辆应该一直在道路中心. 

最后需要限制PID controller的输出范围在[-1,1]之间. 


# Throttle Control
油门控制中需要分为两种情况: 前方有车和前方无车. 

1. 如果前方无车, 则采用定速巡航, 用target speed 和 current speed 的误差作为pid的误差项. 通过调整PID的参数可以得到合适控制器.
2. 如果前方有车, 则控制更为复杂一点. 这里采用了一个串联的PID控制器. 第一个为distance controller, 这个控制器会计算出在当前速度下, 维持0.6倍速度的距离的目标速度. 比如当前速度为50km/h, 则需要与前车维持50* 0.6=30m的距离. 此距离是动态的, 会随着本车的速度的变化而变化.  distance controller会输出target velocity, 然后调用speed controller()得到油门值. 



English Version
This project is divided into three parts: Lane Detection, Steering Control, and Throttle Control. Here's an explanation for each part.

# Lane Detection
## Region Detection
The first step involves cropping the region of interest (ROI) from the camera's image, focusing on the area above the hood and adjacent to both lanes, using `cv::fillpoly` for drawing and `cv::bitwise_and()` for cropping.

## Color Detection
The second step involves color filtering within the ROI area, focusing on white or yellow lane lines using the HSV color space. White lines require high brightness values, while yellow lines need specific hue and saturation ranges. 

It's important to note that reflections, snow, or white objects on the road can interfere with detection. Therefore, using the HSV color space is only suitable for simple road conditions; more complex scenarios may require advanced color filtering methods, such as incorporating deep learning techniques.

## Edge Detection
After isolating white or yellow lane lines, edge detection is applied using `cv::GaussianBlur()` for noise reduction and either `cv::Canny()` or `cv::Sobel()` for edge and corner detection, with Canny generally providing better results.

## Lane Detection
The final step in lane detection involves merging the previously identified edges to form the lanes on each side. 

Initially, `cv::HoughLinesP()` is utilized to detect line segments on edges by fine-tuning parameters within the function, which yields groups of line segments with varying lengths, widths, and distances. By adjusting the parameters, especially increasing the threshold for the length of the segments, it's possible to minimize the impact of small road debris or light reflections, enhancing the accuracy of the detection process.

After obtaining the appropriate group of line segments, it is necessary to distinguish between the left and right lanes, which can be done directly by the range of the slope. Positive slopes are categorized as the right lane, and negative slopes as the left lane. However, it's important to note that merely using the positive or negative distinction may not be sufficient. Lines with very small slopes, such as stop lines or crosswalk lines, could also be incorrectly classified into the left or right lanes, thus interfering with accurate lane detection.

Here's a useful trick: import the slopes of all detected raw lines into a CSV file, manually drive the vehicle through the entire course, and then analyze the slope range in the CSV. In my case, the slope range for the right lane was [0.3, 0.75], and for the left lane, it was [-0.75, -0.2].

After identifying the raw lines for each lane, these segments need to be merged into one. This is achieved using `cv::fitLine()`, which applies the principle of least squares to fit the points into a single line, representing each identified lane.

To enhance the robustness of the results, the historical data average of identified lines is used instead of the currently detected lines. This method significantly improves performance, as relying solely on current detection can lead to minor discrepancies due to the variation in detected line segments with each scan, manifesting as jitter in the recognized lane lines during practical application. 

Finally, the identified lane lines are drawn onto the original image to complete the lane detection process.
# Steering Control
In steering control, a PID controller is used to determine the appropriate steering value. There are several reference values for defining the proportional error P:
1. By identifying the left and right lanes, a line connecting the lane center and vehicle center can be calculated. This line indicates the vehicle's direction, which should remain vertical during normal driving. Thus, the angle between this line and the vertical direction can serve as the proportional error P.
2. The deviation between the vehicle center and the road center can also be calculated based on the identified lanes. Normally, the vehicle should stay centered on the road.

Finally, the output range of the PID controller must be limited to between [-1,1].
# Throttle Control
In throttle control, the approach differs based on whether there's a car ahead:
1. Without a car ahead, cruise control is used, with the PID error term being the difference between target and current speed. Adjusting the PID parameters yields the appropriate controller.
2. With a car ahead, the control becomes more complex, involving a cascaded PID controller. The first is a distance controller, calculating a target speed to maintain a distance of 0.6 times the current speed in meters. For instance, if the current speed is 50 km/h, the vehicle should maintain a distance of 30 meters from the car ahead, calculated as 50 * 0.6. This distance is dynamic, changing with the speed of the vehicle.The distance controller outputs a target velocity, which then determines the throttle value through a speed controller.

Deutsch Version
Dieses Projekt ist in drei Teile gegliedert: Fahrspurerkennung, Lenkung und Gaspedalsteuerung. Hier ist eine Erklärung für jeden Teil.

# Fahrspur-Erkennung
## Erkennung der Region
Der erste Schritt besteht darin, die Region von Interesse (ROI) aus dem Kamerabild auszuschneiden und sich dabei auf den Bereich über der Motorhaube und neben den beiden Fahrspuren zu konzentrieren, wobei `cv::fillpoly` zum Zeichnen und `cv::bitwise_and()` zum Zuschneiden verwendet wird.

## Farberkennung
Der zweite Schritt umfasst die Farbfilterung innerhalb des ROI-Bereichs und konzentriert sich auf weiße oder gelbe Fahrspurlinien unter Verwendung des HSV-Farbraums. Weiße Linien erfordern hohe Helligkeitswerte, während gelbe Linien bestimmte Farbton- und Sättigungsbereiche benötigen. 

Es ist wichtig zu beachten, dass Reflexionen, Schnee oder weiße Objekte auf der Straße die Erkennung beeinträchtigen können. Die Verwendung des HSV-Farbraums eignet sich daher nur für einfache Straßenverhältnisse; komplexere Szenarien erfordern unter Umständen fortschrittlichere Farbfiltermethoden, wie z. B. die Einbeziehung von Deep-Learning-Techniken.

## Kantendetektion
Nach der Isolierung weißer oder gelber Fahrbahnlinien wird die Kantenerkennung mit `cv::GaussianBlur()` zur Rauschunterdrückung und entweder mit `cv::Canny()` oder `cv::Sobel()` zur Kanten- und Eckenerkennung durchgeführt, wobei Canny im Allgemeinen bessere Ergebnisse liefert.

## Lane Detection
Der letzte Schritt der Fahrspurerkennung besteht darin, die zuvor identifizierten Kanten zu den Fahrspuren auf jeder Seite zusammenzufügen. 

Zunächst wird `cv::HoughLinesP()` verwendet, um Liniensegmente an den Kanten zu erkennen, indem die Parameter innerhalb der Funktion fein abgestimmt werden, was zu Gruppen von Liniensegmenten mit unterschiedlichen Längen, Breiten und Abständen führt. Durch die Anpassung der Parameter, insbesondere durch die Erhöhung des Schwellenwerts für die Länge der Segmente, ist es möglich, die Auswirkungen von kleinen Straßenabfällen oder Lichtreflexionen zu minimieren und die Genauigkeit des Erkennungsprozesses zu verbessern.

 Nachdem die entsprechende Gruppe von Liniensegmenten ermittelt wurde, muss zwischen der linken und der rechten Spur unterschieden werden, was direkt über den Bereich der Steigung erfolgen kann. Positive Steigungen werden als rechte Spur und negative Steigungen als linke Spur kategorisiert. Es ist jedoch wichtig zu beachten, dass die bloße Unterscheidung zwischen positiven und negativen Steigungen möglicherweise nicht ausreicht. Linien mit sehr geringen Steigungen, wie z. B. Haltelinien oder Zebrastreifen, könnten ebenfalls fälschlicherweise der linken oder rechten Fahrspur zugeordnet werden und so die genaue Fahrspurerkennung beeinträchtigen.

Hier ein nützlicher Trick: Importieren Sie die Neigungen aller erkannten Rohlinien in eine CSV-Datei, fahren Sie das Fahrzeug manuell durch den gesamten Parcours, und analysieren Sie dann den Neigungsbereich in der CSV-Datei. In meinem Fall war der Neigungsbereich für die rechte Spur [0,3, 0,75] und für die linke Spur [-0,75, -0,2].

Nach der Identifizierung der rohen Linien für jede Fahrspur müssen diese Segmente zu einem einzigen zusammengeführt werden. Dazu wird `cv::fitLine()` verwendet, das das Prinzip der kleinsten Quadrate anwendet, um die Punkte in eine einzige Linie einzupassen, die jede identifizierte Fahrspur repräsentiert.

Um die Robustheit der Ergebnisse zu verbessern, wird der historische Datendurchschnitt der erkannten Linien anstelle der aktuell erkannten Linien verwendet. Diese Methode verbessert die Leistung erheblich, da die alleinige Verwendung der aktuellen Erkennung zu geringfügigen Diskrepanzen führen kann, da die erkannten Liniensegmente bei jedem Scan variieren, was sich in der praktischen Anwendung in Form von Schwankungen der erkannten Fahrspurlinien äußert. 

Abschließend werden die erkannten Fahrspurlinien auf das Originalbild gezeichnet, um den Prozess der Fahrspurerkennung abzuschließen.

# Lenkungsregelung
Bei der Lenkregelung wird ein PID-Regler verwendet, um den geeigneten Lenkwert zu bestimmen. Es gibt mehrere Referenzwerte für die Definition der proportionalen Abweichung P:
1. Durch die Bestimmung der linken und rechten Fahrspur kann eine Linie berechnet werden, die die Fahrspurmitte mit der Fahrzeugmitte verbindet. Diese Linie zeigt die Richtung des Fahrzeugs an, die bei normaler Fahrt senkrecht bleiben sollte. Somit kann der Winkel zwischen dieser Linie und der vertikalen Richtung als proportionaler Fehler P dienen.
2. Die Abweichung zwischen der Fahrzeugmitte und der Straßenmitte kann auch auf der Grundlage der identifizierten Fahrspuren berechnet werden. Normalerweise sollte das Fahrzeug auf der Straße zentriert bleiben.

Schließlich muss der Ausgangsbereich des PID-Reglers auf einen Wert zwischen [-1,1] begrenzt werden.

# Drosselklappensteuerung
Bei der Drosselklappensteuerung hängt die Vorgehensweise davon ab, ob ein Auto vorausfährt:
1. Ohne vorausfahrendes Fahrzeug wird der Geschwindigkeitsregler verwendet, wobei der PID-Fehlerterm die Differenz zwischen Soll- und Ist-Geschwindigkeit ist. Die Anpassung der PID-Parameter ergibt den entsprechenden Regler.
2. Wenn ein Fahrzeug vorausfährt, wird die Regelung komplexer und umfasst einen kaskadierten PID-Regler. Der erste ist ein Abstandsregler, der eine Zielgeschwindigkeit berechnet, um einen Abstand von 0,6 mal der aktuellen Geschwindigkeit in Metern einzuhalten. Beträgt die aktuelle Geschwindigkeit beispielsweise 50 km/h, sollte das Fahrzeug einen Abstand von 30 Metern zum vorausfahrenden Fahrzeug einhalten, berechnet als 50 * 0,6. Dieser Abstand ist dynamisch und ändert sich mit der Geschwindigkeit des Fahrzeugs, wobei der Abstandsregler eine Zielgeschwindigkeit ausgibt, die dann über einen Geschwindigkeitsregler den Drosselwert bestimmt.