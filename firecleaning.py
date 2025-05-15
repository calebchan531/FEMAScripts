// ARFF File Generation for Weka
// Save this as PrepareFireData.java

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;

public class PrepareFireData {
    public static void main(String[] args) throws Exception {
        // Load CSV file
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("Fire.csv"));
        Instances data = loader.getDataStructure();
        data = loader.getDataSet();
        
        System.out.println("Loaded " + data.numInstances() + " instances with " + 
                           data.numAttributes() + " attributes.");
        
        // 1. Convert specific attributes to nominal (especially binary values and targets)
        NumericToNominal numToNom = new NumericToNominal();
        String[] nominalAttOptions = new String[]{"-R", "ihpEligible,sbaEligible,habitabilityRepairsRequired,destroyed,tsaEligible,rentalAssistanceEligible,repairAssistanceEligible,replacementAssistanceEligible,personalPropertyEligible"};
        numToNom.setOptions(nominalAttOptions);
        numToNom.setInputFormat(data);
        data = Filter.useFilter(data, numToNom);
        
        // 2. Remove useless attributes (constant values or with too many missing values)
        RemoveUseless removeUseless = new RemoveUseless();
        removeUseless.setInputFormat(data);
        data = Filter.useFilter(data, removeUseless);
        
        // 3. Replace missing values
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissing);
        
        System.out.println("After preprocessing: " + data.numInstances() + " instances with " + 
                          data.numAttributes() + " attributes.");
        
        // Save the preprocessed data
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("Fire_preprocessed.arff"));
        saver.writeBatch();
        System.out.println("Saved preprocessed data to Fire_preprocessed.arff");
        
        // For each potential target attribute, create a version with that as class
        String[] targets = {"ihpEligible", "habitabilityRepairsRequired", "destroyed", 
                            "tsaEligible", "rentalAssistanceEligible", "personalPropertyEligible"};
        
        for (String target : targets) {
            // Find the index of the target attribute
            int targetIndex = -1;
            for (int i = 0; i < data.numAttributes(); i++) {
                if (data.attribute(i).name().equals(target)) {
                    targetIndex = i;
                    break;
                }
            }
            
            if (targetIndex >= 0) {
                // Create a copy of the data with this attribute as class
                Instances targetData = new Instances(data);
                targetData.setClassIndex(targetIndex);
                
                // Apply SMOTE to balance classes (only if the class is nominal)
                if (targetData.classAttribute().isNominal()) {
                    SMOTE smote = new SMOTE();
                    smote.setInputFormat(targetData);
                    targetData = Filter.useFilter(targetData, smote);
                    
                    System.out.println("After SMOTE for " + target + ": " + 
                                    targetData.numInstances() + " instances");
                }
                
                // Save the target-specific data
                ArffSaver targetSaver = new ArffSaver();
                targetSaver.setInstances(targetData);
                targetSaver.setFile(new File("Fire_" + target + ".arff"));
                targetSaver.writeBatch();
                System.out.println("Saved " + target + " dataset to Fire_" + target + ".arff");
            }
        }
    }
}