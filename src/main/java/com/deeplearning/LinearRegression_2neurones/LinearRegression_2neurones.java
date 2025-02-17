package com.deeplearning.LinearRegression_2neurones;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LinearRegression_2neurones {

	public static void main(String[] args) {
		// Définir un réseau neuronal simple avec une seule couche
		MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
				.seed(123)
				.weightInit(WeightInit.NORMAL)
				.updater(new Adam(0.01))	// Utilisation correcte du taux d'apprentissage
				.l2(1e-4)
				.list()
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)	// Mean Squared Error
						.activation(Activation.IDENTITY)	//activation linéaire
						.nIn(1)
						.nOut(1)
						.build())
				.build());
		
		model.init();
		
		//génération de données d'entraînement
		double[] inTrain = new double[100];
		double[] labelTrain = new double[100];
		for (int i = 0; i < 100; i++) {
		    inTrain[i] = i;
		    labelTrain[i] = 2 * i + 1;  // Toujours basé sur y = 2x + 1
		}
		
		// Données d'entraînement : x = {1, 2, 3, 4}, y = {3, 5, 7, 9}
        // Correspond à y = 2 x + 1
		INDArray input = Nd4j.create(inTrain, new int[] {100, 1});
		INDArray labels = Nd4j.create(labelTrain, new int[] {100, 1});
		
		DataSet dataSet = new DataSet(input, labels);
		
		//Entraînement
		for(int i = 0; i < 8000; i++) {
			model.fit(dataSet);
		    if (i % 500 == 0) {
		        System.out.println("Époque " + i + " - Score: " + model.score());
		    }
		}
		
		//Faire une prédiction
		INDArray testinput = Nd4j.create(new double[] {10, 100}, new int[] {2, 1});
		INDArray output = model.output(testinput);
		
		INDArray weights = model.getParam("0_W");
		INDArray bias = model.getParam("0_b");
		
		System.out.println("Input shape: " + input.shapeInfoToString());
		System.out.println("Labels shape: " + labels.shapeInfoToString());

		// Afficher les résultats
        System.out.println("Prédiction pour x=10 : " + output.getDouble(0));
        System.out.println("Prédiction pour x=100 : " + output.getDouble(1));
        System.out.println("Poids entraîné : " + weights.getDouble(0));
        System.out.println("Biais entraîné : " + bias.getDouble(0));
        
        model.close();
	}
}
