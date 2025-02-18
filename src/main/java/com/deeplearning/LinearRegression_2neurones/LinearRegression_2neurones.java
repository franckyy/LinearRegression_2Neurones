package com.deeplearning.LinearRegression_2neurones;

import java.util.Arrays;

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
						.nOut(2)	//2 neuronnes en sortie
						.build())
				.build());
		
		model.init();
		
		//génération de données d'entraînement
		int nSamples = 100;
		
		double[] inTrain = new double[nSamples];
		double[][] labelTrain = new double[nSamples][2];
		for (int i = 0; i < 100; i++) {
		    inTrain[i] = i;
		    labelTrain[i][0] = 2 * i + 1;  // basé sur y = 2x + 1
		    labelTrain[i][1] = 3 * 1 + 5;	//basé sur y = 3 x + 5
		}
		
		// Données d'entraînement
		INDArray input = Nd4j.create(inTrain, new int[] {nSamples, 1});
		INDArray labels = Nd4j.createFromArray(labelTrain);
		
		DataSet dataSet = new DataSet(input, labels);
		
		//saut de lligne
		System.out.println("");
		
		//Entraînement
		for(int i = 0; i < 8000; i++) {
			model.fit(dataSet);
		    if (i % 500 == 0) {
		        System.out.println("Époque " + i + " - Score: " + model.score());
		    }
		}
		
		//Faire une prédiction
		INDArray testInput = Nd4j.create(new double[] {10, 100}, new int[] {2, 1});
		INDArray output = model.output(testInput);

		//saut de lligne
		System.out.println("");
		
		//Affichages des formes des données
		System.out.println("Output shape: " + Arrays.toString(output.shape()));
		System.out.println("Input shape: " + input.shapeInfoToString());
		System.out.println("Labels shape: " + labels.shapeInfoToString());
		
		//saut de lligne
		System.out.println("");
		
		//Récupération des poids et bias
		INDArray weights = model.getParam("0_W");
		INDArray bias = model.getParam("0_b");
		
		// Afficher les poids et les biais pour chaque neurone
		for (int i = 0; i < weights.columns(); i++) {
		    System.out.println("Poids du neurone " + (i + 1) + " : " + weights.getDouble(0, i));
		    System.out.println("Biais du neurone " + (i + 1) + " : " + bias.getDouble(i));
		}
		
		//saut de lligne
		System.out.println("");
        
        // Afficher les prédictions pour chaque entrée de test et chaque neurone
        for (int i = 0; i < testInput.rows(); i++) {
            double inputVal = testInput.getDouble(i);
            double predictionNeuron1 = output.getDouble(i, 0); // Prédiction du premier neurone
            double predictionNeuron2 = output.getDouble(i, 1); // Prédiction du deuxième neurone
            System.out.println("Pour x = " + inputVal + " : Neurone 1 = " + predictionNeuron1 + ", Neurone 2 = " + predictionNeuron2);
        }
		
		//saut de lligne
		System.out.println("");
        
        // Afficher les résultats
        for (int i = 0; i < input.rows(); i++) {
            double inputVal = input.getDouble(i);
            double outputNeuron1 = output.getDouble(i, 0); // Sortie du premier neurone
            double outputNeuron2 = output.getDouble(i, 1); // Sortie du deuxième neurone
            System.out.println("Pour x = " + inputVal + " : Neurone 1 = " + outputNeuron1 + ", Neurone 2 = " + outputNeuron2);
        }
        model.close();
	}
}
