(ns dl4j-clj-example.core
   (:import [org.deeplearning4j.datasets.iterator DataSetIterator SamplingDataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl IrisDataSetIterator]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration Updater
            NeuralNetConfiguration NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.conf.layers OutputLayer OutputLayer$Builder
            RBM RBM$Builder RBM$HiddenUnit RBM$VisibleUnit]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.params DefaultParamInitializer]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.api IterationListener]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset SplitTestAndTrain]
           [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
           [java.nio.file Files]
           [java.nio.file Paths]
           [java.util Arrays]
           [java.util Random]))
         
(defn setup []
  (set! Nd4j/MAX_SLICES_TO_PRINT -1)
  (set! Nd4j/MAX_ELEMENTS_PER_SLICE -1)
  (set! Nd4j/ENFORCE_NUMERICAL_STABILITY true))

(defn neural-net-configuration-builder []
  (NeuralNetConfiguration$Builder.))

(defn iris-dataset-iterator [batch-size num-samples]
  (let [iter (-> (IrisDataSetIterator. batch-size num-samples)
                 (.next))]
    (.normalizeZeroMeanZeroUnitVariance iter)
    (.shuffle iter)
    iter))

(defn split-test-and-train [dataset-iter split-train-num seed]
  (let [test-and-train (.splitTestAndTrain dataset-iter
                                           split-train-num
                                           (Random. (int seed)))]
    [(.getTrain test-and-train)
     (.getTest test-and-train)]))

(defn build-nn-cfg1 [seed iterations num-rows num-columns output-num]
  (-> (neural-net-configuration-builder)
      (.seed seed)
      (.iterations iterations)
      (.learningRate 1E-1)
      (.optimizationAlgo OptimizationAlgorithm/CONJUGATE_GRADIENT)
      (.l1 1E-1)
      (.regularization true)
      (.l2 2E-4)
      (.useDropConnect true)
      (.list 2)
      (.layer 0 (-> (RBM$Builder. RBM$HiddenUnit/RECTIFIED
                                  RBM$VisibleUnit/GAUSSIAN)
                    (.nIn (* num-rows num-columns))
                    (.nOut 3)
                    (.weightInit WeightInit/XAVIER)
                    (.k 1)
                    (.activation "relu")
                    (.lossFunction LossFunctions$LossFunction/RMSE_XENT)
                    (.updater Updater/ADAGRAD)
                    (.dropOut 0.5)
                    (.build)))
      (.layer 1 (-> (OutputLayer$Builder. LossFunctions$LossFunction/MCXENT)
                    (.nIn 3)
                    (.nOut output-num)
                    (.activation "softmax")
                    (.build)))
      (.build)))

(defn create-multi-layer-network [cfg]
  (let [model (MultiLayerNetwork. cfg)
        _ (.init model)]
    model))

(defn rnn-test []
  (let [;; values
        num-rows 4
        num-columns 1
        output-num 3
        num-samples 150
        batch-size 150
        iterations 300
        split-train-num (int (* batch-size 0.8))
        seed 111
        listener-freq (/ iterations 5)

        ;; load data
        data-iter (iris-dataset-iterator batch-size num-samples)

        ;; split data
        [train tst] (split-test-and-train data-iter split-train-num seed)

        ;; build model
        conf (build-nn-cfg1 seed iterations num-rows num-columns output-num)

        ;; create model
        model (create-multi-layer-network conf)

        ;;_ (.setListeners model (to-array [(ScoreIterationListener. listener-freq)]))
        _ (.fit model train)

        ;; evaluate and log
        evaluation (Evaluation. output-num)
        output (.output model (.getFeatureMatrix tst))]
    (dotimes [i (.rows output)]
      (let [actual (-> tst
                       .getLabels
                       (.getRow i)
                       .toString
                       .trim)
            predicted (-> output
                          (.getRow i)
                          .toString
                          .trim)]
        (println [actual predicted])))
    (.eval evaluation (.getLabels tst) output)
    (println (.stats evaluation))))

(defn -main [& args]
  (rnn-test))
