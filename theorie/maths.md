## sigmoid-funktion:
link = https://www.digitalocean.com/community/tutorials/sigmoid-activation-function-python 
- das resultat der sigmoidfunktin wird genutzt um einem wert eine bedeutung zu geben und diese bedeutng im rahmen zu ehalten damit das ganze übersichtlich bleibt. es kann als licht verstanden werden, je heller desto wichtiger ist das merkmal für eine gute prediction!
- mit den weights wird diese helligkeit angepasst und verändert!
- helligkeit = weight * inputwert oder output wert des letzten layers (= activations) -bias
- der bias sagt wie hoch das resultat sein muss der sigmoid-funktion damit es meanignful ist!
- der wert wird erst durch die sigmoid-fntion gelassen wenn dieser mit den weights multipliziert ist ansonsten können die resultate irgendeine grösse haben ausserhalb des bereichs den wir wollen!

neuronen = zahlen = funktionswerte einer sigmoid-funktion = outputs der sigmoid-funktion = die wert die durch da netz wandern und ständig verändert werden

- forward-propagation = eine prediction machen --> der input muss immer angepasst werden mit der sigmoid funktion --> mithilfe der weights und der sigmoidfunktion wird der iinput forlaufend angepasst
- backward-propagatio = weights anpassen damit der output dem desired output mehr entspticht --> die weights müssen immer angepast werden mit ... --> mithilfe des outputs der predistion und der funktion ... werden di eeights neu berechnent

## fragen
- wiso kommen beim input von sehr hohen werten immer eine 1 heraus..bereits ab 80 etc
- wie macht das netzerk eine prediciton ..welche berechnungen erfolgen?
- was ist der unterschied zischen bias ans weights
- wie kann man die benötigten parameters berechnen?
- ist das resultat einer sigmoid-funktion eine zahl oder eine matrix etc...
- wir die cost-function für eine prediction berechnet oder für das gesamte dataset?
- wie werden semantische vektoren gemacht?
- müsen alle hidden layers gleich viele neuronen haben?
- ist das resultat der cost-funtion eine zahl pro neuron oder eher einen gradienten?


## Train the Network
1. data goes through the network -> from input to a prediction
   1. the networ is divided into 3 parts -> input layer + hidden layers + output layer -> data + function which processes the data + prediction 
   2. tha raw data enters the network in a vektorform. each inputneuron goes through the sigmoid funktion where its value gets changed.input und output habe diesselbe form nur ist  (input= tensor mit alle activations des lvorherigen layers, output = tensor mit allen werten für jede activation jedes neuron)
2. prediction is evaluated, how good can the network predict?
   1. cost-funktion = sum of all predictions of the utput layer - sum of all right answers for the output layer given a certain input. als input wierden hier alle parameter laso weights und biases genommen
   2. wenn man das gesamte netzwerk als eine funktion betrachtet  wobei der output der der cost-funktion ist dann muss man das globale minimum finden und dabei auch die inputs der cost-funktion so verändern dass dieses minimum erreicht wird! mit inputs sind hier alle weights gemeint die man dann anpassen muss!
   3. die cost funktion gibt eine zahl (abweichung der prediction vom desired-wert) heraus und einen gradient (welche weights sind wie wichtig?)
3. the network is corrected so that the prediction will be better

netzwerk trainieren = weights anpassen