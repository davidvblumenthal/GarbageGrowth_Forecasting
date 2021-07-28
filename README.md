# Forecasting recycling glass growth in the city of Frankfurt

## Kurze Projekt Beschreibung

### 🎯 Business Unterstanding

Die Frankfurter  Entsorgungs- und Service GmbH hat 72 Glascontainer in Frankfurt mit entsprechender Sensorik ausgestattet. Hierbei werden Daten zum Füllstand, der Temperatur im Container sowie Neigungswinkel gesammelt.

Das Projekt befindet sich aktuell noch in einer Explorationsphase was in Folge bedeutet, dass es noch keinen standardisierten Ansatz gibt diese zu verwerten.

Mithilfe der Daten soll ein effizienteres leeren der Container ermöglicht werden. Es gilt somit ein unterstützendes Prediktionsmodell zu entwickeln, welches Aussagen über das Füllverhalten der Container treffen kann.

## 📊 Data Understanding

Bei den Daten handelt es sich um Zeitreihen, die aus der Beobachtung einzelner Variablen über einen Zeitraum entstehen. Die entscheide Variable **Höhe** wird dabei unterschiedlich oft beobachten allerdings im Schnitt mehrmals pro Tag. Da es sich dabei allerdings um EDGE Geräte handelt, sind einzelne Messungen nicht immer akkurat und zusätzlich sind häufig fehlende Daten zu beobachten. So können die Messungen für unterschiedliche Container stark varieren. Im folgenden Bild ist die Füllhöhe eines Containers über den bereitgestellten Zeitraum zu beobachten.

![Untitled%20dc564a8fe174488583651b8e230af1de/brauchbar.png](doku_ressources/brauchbar.png)

[DataVisualisation.ipynb](https://git.scc.kit.edu/uflgi/bda-analytics-challenge-template/-/blob/master/notebooks/DataVisualization.ipynb)

Um Einflüsse auf den Füllstand besser zu verstehen, haben wir uns unterschiedliche Datenquellen aus dem Internet besorgt und diese mithilfe einer Korrelationsanalyse auf ihren Einfluss überprüft.

Untersuchte Datenquellen:

☁️  Wetterdaten - Intuition: Tendenziell wird Glas eher bei gutem Wetter entsorgt als bei Regen. Wer verlässt schon gerne bei Regen das Haus?

🎉  Feiertage - Intuition: Über Feiertage kommen Menschen gerne zusammen, was zu einem vermehrten aufkommen Restglas führen könnte (Weinflaschen etc.)

⚽  Fußballspiele & Großveranstaltungen - Intuition: Zu Fußballspielen und anderen Großveranstaltungen ist das Menschenaufkommen in der Stadt erhöht. Auch hier wird wieder vermehrt Alkohol konsumiert und evtl. direkt in auf dem Weg liegende Glascontainern entsorgt.

[webscraper.ipynb](https://git.scc.kit.edu/uflgi/bda-analytics-challenge-template/-/blob/master/notebooks/webscraper.ipynb)

## 📈  Data Preparation

Da Qualität der Vorhersagen des Modells direkt mit der Qualität der Trainingsdaten zusammenhängt, galt es die Daten möglichst gut für das Training vorzubereiten.

Nachdem alle Containerfüllstände visualisiert wurden, wurden Container, deren Datenqualität extrem fragwürdig waren über eine deskriptive Analyse aussortiert. In den folgenden Plots sind jeweils ein Beispiel für einen Aussortierten (unten) und einen Weiterverwendeten (oben) zu sehen.

![Untitled%20dc564a8fe174488583651b8e230af1de/brauchbar%201.png](doku_ressources/brauchbar%201.png)

![Untitled%20dc564a8fe174488583651b8e230af1de/unbrauchbar.png](doku_ressources/unbrauchbar.png)

In den weiteren Schritten wurden offensichtliche Messfehler in den Container entfernt z. B. Höhe > 190. Da in den Daten weiterhin sehr viel Noise enthalten ist galt es, denn unterliegenden Trend zu extrahieren. Hierbei wurden unterschiedliche Smoothing Verfahren angewandt. Im Code Snippet sowie in der Grafik unten wurde jeweils das Smoothing Verfahren des Rolling Average mit einer Fenstergröße von 30 visualisiert.

```python
df['mov_avg'] = df['Height'].rolling(30).mean()
```

![Untitled%20dc564a8fe174488583651b8e230af1de/plot.png](doku_ressources/plot.png)

Außerdem haben wir die Daten, nach dem erste Modellierungsversuche scheiterten, auf Tagesniveau mithilfe mean(), min() und max() Operators aggregiert. Dabei waren die Ergebnisse des mean() näher am Rohverlauf als die anderen, weshalb für das weitere Vorgehen diese Daten verwendet wurden. 
Da bei den meisten Containern nicht für jeden Tag Messwerte existierten, haben wir fehlende Messwerte mithilfe von Interpolationsverfahren aufgefüllt.

### 🌀 Clustering

Da sich bei der deskriptiven Analyse zeigte, dass es visuelle Ähnlichkeiten zwischen den Zeitreihen gibt, wollte wir diese zu Clustern zusammenfassen und für jedes Cluster ein Modell trainieren um so die Güte der einzelnen und damit die gesamt Vorhersagequalität zu erhöhen. 

Dafür wurde Dynamic Time Warping in Kombination mit K-means Clustering eingesetzt. Dafür wurde die Bibliothek [tslearn](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html) verwendet. Dabei hat sich **k = 3** als effektivste Anzahl für die Cluster herauskristallisiert.

![Untitled%20dc564a8fe174488583651b8e230af1de/clusters.png](doku_ressources/clusters.png)

![Untitled%20dc564a8fe174488583651b8e230af1de/Screenshot_2021-07-24_at_04.05.14.png](doku_ressources/Screenshot_2021-07-24_at_04.05.14.png)

[preprocessing_clustering.ipynb](https://git.scc.kit.edu/uflgi/bda-analytics-challenge-template/-/blob/master/notebooks/preprocessing_clustering.ipynb)

[data_preprocessing.ipynb](https://git.scc.kit.edu/uflgi/bda-analytics-challenge-template/-/blob/master/notebooks/data_preprocessing.ipynb)

[mapsVizualization.ipynb](https://git.scc.kit.edu/uflgi/bda-analytics-challenge-template/-/blob/master/notebooks/mapsVizualization.ipynb)

## 🤖 Modelling

### Lineare Regression als Baseline

❗ Die wesentliche Problem für das Modellieren sind die eher schlechte Datenqualität sowie die wechselnden Standorte der Container.

Dabei macht es keinen Sinn die eigentlichen Kurven mit ihren jeweiligen Leerungsintervallen zu leeren, da ja gerade diese angepasst werden sollen.

Als Baseline haben wir versucht dies mithilfe einer einfachen linearen Regression zu modellieren. Dabei wurde jeweils aus einer Zeitreihe pro Cluster ein Füllintervall extrahiert und darauf die Steigung berechnet. Dies führte für diese recht einfache Methode auch zu sehr vielversprechenden Ergebnissen. Um die Ergebnisse zu evaluieren, haben wir versucht, die jeweiligen Leerungen mithilfe einer Schwellwertfunktion zu berechnen. Dies funktioniert auf unterschiedlichen Zeitreihen mit stark schwankendem Erfolg.

![Untitled%20dc564a8fe174488583651b8e230af1de/Screenshot_2021-07-24_at_17.14.01.png](doku_ressources/Screenshot_2021-07-24_at_17.14.01.png)

![Untitled%20dc564a8fe174488583651b8e230af1de/Untitled.png](doku_ressources/Untitled.png)

Das Verfahren sieht auf den ersten Blick tatsächlich sehr vielsprechend aus, kann aber nicht optimal evaluiert werden, da in den Daten die Zeitpunkte der Leerung garnicht, bis sehr schlecht angegeben sind.

**Hierfür wären Daten notwendig, die die korrekten Leerungszeitpunkte enthalten.** 

[linear_regression.ipynb](https://git.scc.kit.edu/uflgi/bda-analytics-challenge-template/-/blob/master/notebooks/performing_linear_regression.ipynb)

[linearRegression.ipynb](https://git.scc.kit.edu/uflgi/bda-analytics-challenge-template/-/blob/master/notebooks/linearRegression.ipynb)

### LSTM

Drei Ansätze für ein LSTM wurden verfolgt: 
    1.   LSTM auf Basis eines Clusters mit Daten für jede Stunde (Notebook: Cluster0_LSTM.ipynb, Modell: 
    2.   LSTM auf Basis aller Cluster mit Daten für jede Stunde (Notebook: Overall_LSTM.ipynb, Modell: 
    3.   LSTM auf Basis aller Cluster mit Daten für jeden Tag (Notebook: Overall_grouped_LSTM.ipynb, Modell: 
    
Nur Ansatz 3. erzeugt passende Vorhersagen: 

![Untitled%20dc564a8fe174488583651b8e230af1de/Overall_pred.png](doku_ressources/Overall_pred.png)

Hierbei lernt das LSTM das Füllverhalten aller Container. Die Leerungen nehmen wir manuell vor, indem die Input Werte für das LSTM auf einen leeren Container zurückgesetzt werden, sobald der Füllstand bei mehreren Vorhersagen gleich bleibt.

Dieser Ansatz wäre mit zusätzlichen Füllstandsdaten auch für einzelne Container möglich.

## 🚀 Fazit/Future Work

Die Lineare Regression trifft die Entwicklung des Füllstands in vielen Fällen bereits sehr gut, was dafür spricht, dass der Füllstand keiner komplexen Funktion folgt und daher in der Tendenz ein relativ einfach zu modellierender Trend zugrunde liegt.

❗ Hierbei wurden allerdings keine Ausreißer Füllstände mit in Betracht gezogen. Um diese zu modellieren ist die Kapazität einer einfachen Regression nicht hoch genug. 

Der Engpass für die Evaluation ist hier klar der korrekte Zeitpunkt der Leerung. Mit unserer einfachen Schwellwertfunktion sind wir nicht sicher in der Lage diese zu genau genug zu approximieren.

→ Zwar sagt die Sensor Dokumentation, dass diese enthalten seien sollten, dies konnten wir allerdings in den Daten nicht beobachten. Hierfür kann entweder der Sensor adjustiert werden oder um aus diesen Daten mehr zu machen, wäre unser Vorschlag einen Teil der Daten manuell zu Labeln und dann über ein Supervised Learning Verfahren ein Modell zu trainieren, dass in der Lage ist die Leerungszeitpunkte genauer anzugeben. Leider kam uns diese Idee zu spät und konnte im Rahmen dieses Projekts leider nicht mehr umgesetzt werden.



## Special Stuff
Our LSTM model was trained on Google Colab. The data split and model training is not performed in the notebooks, but 
the processed is visualized. The whole functionality can be found in the src folder.


## Project Organization
------------
```
	├── README.md 							<-- this file. insert group members here
	├── .gitignore 						    <-- prevents you from submitting several clutter files
	├── data
	│   ├── modeling
	│   │   ├── dev 						<-- your development set goes here
	│   │   ├── test 						<-- your test set goes here
	│   │   └── train 						<-- your train set goes here goes here
	│   ├── preprocessed 					<-- your preprocessed data goes here
	│   └── raw								<-- the provided raw data for modeling goes here
	├── docs								<-- provided explanation of raw input data goes here
	│
	├── models								<-- dump models here
	├── notebooks							<-- your playground for juptyer notebooks
	├── requirements.txt 					<-- required packages to run your submission (use a virtualenv!)
	└── src
	    ├── additional_features.py 			<-- your creation of additional features/data goes here
	    ├── cluster.py 						<-- code for clustering
	    ├── regression.py 					<-- regression model
	    ├── preprocessing.py 				<-- your preprocessing script goes here
	    ├── train.py 						<-- your training script goes here
	    └── predict.py 						<-- prediction script for LSTM
	
```

